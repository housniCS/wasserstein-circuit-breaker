"""
metrics.py
Calculateur vectorisé des métriques financières et statistiques :
1. Log-rendements : r_t = ln(P_t / P_{t-1})
2. Volatilité de Garman-Klass (exploitant Open, High, Low, Close)
3. Divergence de Wasserstein 1D glissante (W1 entre fenêtre de référence et fenêtre courante)
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def compute_log_returns(series: pd.Series) -> pd.Series:
    """
    Calcule les log-rendements : r_t = ln(P_t / P_{t-1})
    """
    return np.log(series / series.shift(1))


def compute_garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Calcule l'estimateur de volatilité locale de Garman-Klass (1980) :
    sigma_{GK, t}^2 = 0.5 * [ln(H_t / L_t)]^2 - (2 * ln(2) - 1) * [ln(C_t / O_t)]^2
    sigma_{GK, t} = sqrt(max(0, sigma_{GK, t}^2))
    """
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    variance = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    # Sécurité numérique : borne inférieure à 0 pour éviter sqrt(valeur négative)
    variance_clipped = np.maximum(variance, 0.0)
    return np.sqrt(variance_clipped)


def compute_rolling_wasserstein_1d(
    returns: pd.Series,
    ref_window: int = 200,
    curr_window: int = 30
) -> pd.Series:
    """
    Mesure la distance 1D de Wasserstein (Earth Mover's Distance) entre :
    - Fenêtre de référence W_ref : [t - ref_window : t - curr_window] (dynamique de fond)
    - Fenêtre courante W_curr : [t - curr_window : t] (dynamique immédiate)

    Complexité O(K log K) par étape via scipy.stats.wasserstein_distance.
    """
    n = len(returns)
    w1_values = np.full(n, np.nan)
    ret_arr = returns.values

    # On commence dès que l'on dispose d'assez d'historique pour la fenêtre de référence
    for i in range(ref_window, n):
        ref_sample = ret_arr[i - ref_window : i - curr_window]
        curr_sample = ret_arr[i - curr_window : i]
        
        # Vérification qu'il n'y a pas de NaN dans les tranches
        if not (np.isnan(ref_sample).any() or np.isnan(curr_sample).any()):
            w1_values[i] = wasserstein_distance(ref_sample, curr_sample)

    return pd.Series(w1_values, index=returns.index, name="w1_dist")


def compute_vectorized_features(
    df: pd.DataFrame,
    ref_window: int = 200,
    curr_window: int = 30
) -> pd.DataFrame:
    """
    Assemble l'ensemble des caractéristiques vectorisées sur le DataFrame OHLCV :
    - 'ret' : log-rendements
    - 'gk_vol' : volatilité Garman-Klass
    - 'w1' : distance de Wasserstein glissante
    """
    df_out = df.copy()
    df_out["ret"] = compute_log_returns(df_out["close"])
    df_out["gk_vol"] = compute_garman_klass_volatility(df_out)
    df_out["w1"] = compute_rolling_wasserstein_1d(
        df_out["ret"],
        ref_window=ref_window,
        curr_window=curr_window
    )
    return df_out
