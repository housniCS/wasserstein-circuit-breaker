"""
primary_model.py
Modèles primaires directionnels et calcul des résidus d'erreur binaires :
- Classe abstraite DirectionalForecaster
- MomentumForecaster : baseline / proxy rapide
- ChronosBoltForecaster : Foundation Model Time-Series (Amazon Chronos-Bolt) avec batching vectorisé
- compute_directional_target_and_residuals : calcul de e_t = II(y_hat_t != y_{t+1})
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class DirectionalForecaster(ABC):
    """
    Interface pour tout modèle prédictif primaire produisant un signal directionnel {-1, +1}.
    """
    @abstractmethod
    def predict_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Produit un vecteur de signaux directionnels y_hat_t dans {-1, +1}.
        Doit garantir l'absence stricte de biais d'anticipation (lookahead bias).
        """
        pass


class MomentumForecaster(DirectionalForecaster):
    """
    Proxy de modèle primaire directionnel basé sur le momentum court-terme (ex: moyenne 3h).
    """
    def __init__(self, window: int = 3):
        self.window = window

    def predict_direction(self, df: pd.DataFrame) -> pd.Series:
        mom = df["ret"].rolling(self.window).mean()
        pred = np.where(mom >= 0, 1, -1)
        return pd.Series(pred, index=df.index, name="pred_dir")


class ChronosBoltForecaster(DirectionalForecaster):
    """
    Modèle de fondation Amazon Chronos-Bolt (Zero-Shot Time-Series Foundation Model).
    Prévoit la distribution du cours futur P_{t+1} à partir de la fenêtre passée [t - L : t].
    Signal :
        y_hat_t = +1 si P_median_{t+1} >= P_t sinon -1
    """
    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        context_length: int = 48,
        batch_size: int = 64,
        device: str = "cpu",
        cache_path: str = "data/chronos_preds.csv"
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.cache_path = cache_path
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            import torch
            from chronos import ChronosBoltPipeline
            print(f"[Chronos-Bolt] Chargement du modèle de fondation '{self.model_name}'...")
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=torch.float32
            )
        return self._pipeline

    def predict_direction(self, df: pd.DataFrame) -> pd.Series:
        # Vérifier si les prédictions sont déjà en cache pour accélérer
        if self.cache_path and os.path.exists(self.cache_path):
            cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            if len(cached) == len(df) and (cached.index == df.index).all():
                print(f"[Chronos-Bolt] Chargement des prédictions depuis le cache : {self.cache_path}")
                return cached["pred_dir"]

        import torch
        pipeline = self._get_pipeline()
        close_prices = df["close"].values
        n = len(close_prices)
        preds = np.zeros(n, dtype=int)

        print(f"[Chronos-Bolt] Inférence Zero-Shot par batchs (L={self.context_length}, batch={self.batch_size})...")

        # Construction des fenêtres contextuelles
        valid_indices = []
        contexts = []
        last_prices = []

        for i in range(self.context_length, n):
            ctx = close_prices[i - self.context_length : i]
            contexts.append(ctx)
            last_prices.append(close_prices[i - 1])
            valid_indices.append(i)

        # Inférence par batchs pour maximiser les performances
        total_batches = (len(contexts) + self.batch_size - 1) // self.batch_size
        all_pred_directions = []

        for b in range(total_batches):
            b_start = b * self.batch_size
            b_end = min((b + 1) * self.batch_size, len(contexts))
            batch_ctx = contexts[b_start:b_end]
            batch_last_p = last_prices[b_start:b_end]

            # Shape: (B, context_length)
            tensor_ctx = torch.tensor(np.array(batch_ctx), dtype=torch.float32)

            with torch.no_grad():
                # Forecast shape: (B, num_quantiles, prediction_length=1)
                forecast = pipeline.predict(tensor_ctx, prediction_length=1)

            # Extraction de la médiane (quantile central, index 4 dans les 9 quantiles)
            if forecast.ndim == 3 and forecast.shape[1] >= 5:
                # quantiles: [0.1, 0.2, ..., 0.5 (index 4), ..., 0.9]
                median_forecast = forecast[:, 4, 0].cpu().numpy()
            else:
                median_forecast = torch.quantile(forecast, 0.5, dim=1).squeeze(-1).cpu().numpy()

            # Direction : +1 si hausse prévue par rapport au dernier cours connu, sinon -1
            b_dirs = np.where(median_forecast >= np.array(batch_last_p), 1, -1)
            all_pred_directions.extend(b_dirs)

        for idx, direction in zip(valid_indices, all_pred_directions):
            preds[idx] = direction

        # Pour les indices < context_length, fallback momentum 1
        for i in range(self.context_length):
            preds[i] = 1 if i == 0 or close_prices[i] >= close_prices[i - 1] else -1

        res_series = pd.Series(preds, index=df.index, name="pred_dir")

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            res_series.to_frame().to_csv(self.cache_path)
            print(f"[Chronos-Bolt] Prédictions sauvegardées dans : {self.cache_path}")

        return res_series


def compute_directional_target_and_residuals(
    df: pd.DataFrame,
    forecaster: DirectionalForecaster
) -> pd.DataFrame:
    """
    Applique le modèle primaire et calcule le résidu d'erreur binaire :
    1. y_hat_t : prédiction faite à l'instant t
    2. target_dir : sens réel du mouvement futur à l'instant t+1 : sign(r_{t+1})
    3. model_error (e_t) : indicatrice d'erreur II(y_hat_t != target_dir)
    """
    df_out = df.copy()

    # 1. Inférence directionnelle à l'instant t
    df_out["pred_dir"] = forecaster.predict_direction(df_out)

    # 2. Cible réelle future : direction du rendement à t+1
    next_return = df_out["ret"].shift(-1)
    target_dir = np.where(next_return >= 0, 1, -1)
    df_out["target_dir"] = np.where(next_return.isna(), np.nan, target_dir)

    # 3. Résidu binaire d'erreur e_t : 1 si le modèle échoue, 0 s'il a raison
    error = (df_out["pred_dir"] != df_out["target_dir"]).astype(float)
    df_out["model_error"] = np.where(df_out["target_dir"].isna(), np.nan, error)

    return df_out
