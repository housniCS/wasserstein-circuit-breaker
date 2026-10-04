"""
test_stage1.py
Script de test et de validation de la première étape :
1. Chargement des données OHLCV 1h (Binance / Cache)
2. Calcul vectorisé (Log-ret, Garman-Klass, Wasserstein 1D)
3. Comparaison des modèles primaires :
   - Momentum Baseline (3h)
   - Amazon Chronos-Bolt (Foundation Model Time-Series Zero-Shot)
4. Calcul des résidus d'erreur binaires e_t
"""

import sys
import pandas as pd
from src.data_loader import load_ohlcv
from src.metrics import compute_vectorized_features
from src.primary_model import (
    MomentumForecaster,
    ChronosBoltForecaster,
    compute_directional_target_and_residuals
)


def run_stage_1():
    print("=" * 75)
    print(" ÉTAPE 1 : DATA PIPELINE, FEATURES VECTORISÉES & COMPARAISON DES MODÈLES")
    print("=" * 75)

    # 1. Chargement des données
    print("\n[1/4] Chargement des barres OHLCV 1h...")
    df_raw = load_ohlcv(symbol="BTCUSDT", cache_path="data/btc_1h.csv")
    print(f"      Nombre total de barres chargées : {len(df_raw)}")
    print(f"      Plage temporelle : {df_raw.index[0]}  --->  {df_raw.index[-1]}")

    # 2. Calcul des métriques vectorisées
    print("\n[2/4] Calcul des métriques statistiques vectorisées...")
    df_features = compute_vectorized_features(df_raw, ref_window=200, curr_window=30)

    # 3. Modèle Primaire 1 : Proxy Momentum (3h)
    print("\n[3/4] Inférence Modèle 1 : Momentum Baseline (3h)...")
    mom_forecaster = MomentumForecaster(window=3)
    df_mom = compute_directional_target_and_residuals(df_features, mom_forecaster)

    # 4. Modèle Primaire 2 : Amazon Chronos-Bolt (Foundation Model Zero-Shot)
    print("\n[4/4] Inférence Modèle 2 : Amazon Chronos-Bolt (Zero-Shot Time-Series)...")
    chronos_forecaster = ChronosBoltForecaster(
        model_name="amazon/chronos-bolt-tiny",
        context_length=48,
        batch_size=64,
        cache_path="data/chronos_preds.csv"
    )
    df_chronos = compute_directional_target_and_residuals(df_features, chronos_forecaster)

    # Nettoyage et synchronisation pour évaluation sur la même période
    df_valid = df_features.dropna(subset=["w1", "gk_vol"]).copy()
    
    # Intégration des prédictions et résidus des deux modèles
    df_valid["pred_mom"] = df_mom.loc[df_valid.index, "pred_dir"]
    df_valid["err_mom"] = df_mom.loc[df_valid.index, "model_error"]

    df_valid["pred_chronos"] = df_chronos.loc[df_valid.index, "pred_dir"]
    df_valid["err_chronos"] = df_chronos.loc[df_valid.index, "model_error"]
    df_valid["target_dir"] = df_mom.loc[df_valid.index, "target_dir"]

    # Suppression de la dernière ligne (target future inconnue)
    df_eval = df_valid.dropna(subset=["err_mom", "err_chronos", "target_dir"]).copy()
    df_eval["err_mom"] = df_eval["err_mom"].astype(int)
    df_eval["err_chronos"] = df_eval["err_chronos"].astype(int)

    # ------------------------------------------------------------------
    # TABLEAU COMPARATIF DES RÉSULTATS
    # ------------------------------------------------------------------
    hit_ratio_mom = (1 - df_eval["err_mom"].mean()) * 100
    hit_ratio_chronos = (1 - df_eval["err_chronos"].mean()) * 100

    err_rate_mom = df_eval["err_mom"].mean() * 100
    err_rate_chronos = df_eval["err_chronos"].mean() * 100

    # Accord entre les deux modèles
    agreement = (df_eval["pred_mom"] == df_eval["pred_chronos"]).mean() * 100

    print("\n" + "=" * 75)
    print(" COMPARAISON STATISTIQUE : MOMENTUM vs AMAZON CHRONOS-BOLT")
    print("=" * 75)
    print(f"{'Métrique':<35} | {'Momentum (3h)':<16} | {'Chronos-Bolt (Amazon)':<16}")
    print("-" * 75)
    print(f"{'Précision Directionnelle (Hit-Ratio)':<35} | {hit_ratio_mom:>14.2f} % | {hit_ratio_chronos:>14.2f} %")
    print(f"{'Taux d Erreur Brut (e_t = 1)':<35} | {err_rate_mom:>14.2f} % | {err_rate_chronos:>14.2f} %")
    print(f"{'Nombre d Heures Évaluées':<35} | {len(df_eval):>14d}   | {len(df_eval):>14d}  ")
    print(f"{'Corrélation des Signaux (Accord)':<35} | {agreement:>14.2f} % (les deux modèles convergent)")
    print("-" * 75)

    print("\n" + "-" * 75)
    print(" APERÇU DES 5 DERNIÈRES PRÉDICTIONS COMPARÉES")
    print("-" * 75)
    cols = ["close", "gk_vol", "w1", "pred_mom", "pred_chronos", "target_dir", "err_chronos"]
    print(df_eval[cols].tail(5).to_string())

    print("\n" + "=" * 75)
    print(" [CONFIRMATION] Les résidus e_t de Chronos-Bolt sont calculés.")
    print(" L'arbre substitut peut désormais s'entraîner directement sur les résidus d'Amazon Chronos !")
    print("=" * 75)


if __name__ == "__main__":
    run_stage_1()
