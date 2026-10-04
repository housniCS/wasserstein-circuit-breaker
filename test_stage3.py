"""
test_stage3.py
==============
Script de test et de validation de l'Étape 3 :
1. Pipeline complet : Data -> Features -> Modèle Primaire -> Wasserstein Circuit-Breaker
2. Backtest Vectorisé Net de Frais (10 bps par transaction)
3. Comparaison statistique des 3 approches :
   - Stratégie Brute (sans filtre)
   - Stratégie Filtrée (+ Wasserstein Circuit-Breaker)
   - Benchmark Buy & Hold BTC
"""

import pandas as pd
from src.data_loader import load_ohlcv
from src.metrics import compute_vectorized_features
from src.primary_model import ChronosBoltForecaster, compute_directional_target_and_residuals
from src.circuit_breaker import WassersteinCircuitBreaker, apply_circuit_breaker
from src.backtest import compare_strategies


def run_stage_3():
    print("=" * 80)
    print(" ÉTAPE 3 : VALIDATION DU BACKTEST VECTORISÉ AVEC AMAZON CHRONOS-BOLT (10 BPS)")
    print("=" * 80)

    # 1. Chargement & Métriques
    print("\n[1/4] Pipeline amont : Données OHLCV 1h & Features vectorisées...")
    df_raw = load_ohlcv(symbol="BTCUSDT", cache_path="data/btc_1h.csv")
    df_features = compute_vectorized_features(df_raw, ref_window=200, curr_window=30)

    # 2. Modèle Primaire : Amazon Chronos-Bolt (Zero-Shot Time-Series Foundation Model)
    print("\n[2/4] Modèle primaire Amazon Chronos-Bolt & Résidus d'erreur e_t...")
    forecaster = ChronosBoltForecaster(
        model_name="amazon/chronos-bolt-tiny",
        context_length=48,
        batch_size=64,
        cache_path="data/chronos_preds.csv"
    )
    df_stage1 = compute_directional_target_and_residuals(df_features, forecaster)

    # 3. Arbre Substitut (Coupe-Circuit)
    print("\n[3/4] Entraînement et inférence du WassersteinCircuitBreaker...")
    breaker = WassersteinCircuitBreaker(
        max_depth=2,
        min_samples_leaf=50,
        error_threshold=0.52,
        feature_names=["w1", "gk_vol"]
    )
    breaker.fit(df_stage1, target_col="model_error")
    df_stage2 = apply_circuit_breaker(df_stage1, breaker)

    # 4. Backtest Vectorisé & Comparaison
    print("\n[4/4] Exécution du backtest vectorisé net de frictions (10 bps)...")
    equity_df, metrics_df = compare_strategies(
        df_stage2,
        pred_col="pred_dir",
        breaker_col="circuit_breaker",
        ret_col="ret",
        cost_bps=10.0
    )

    print("\n" + "=" * 80)
    print(" RÉSULTATS COMPARATIFS DU BACKTEST (NET DE FRAIS : 10 BPS)")
    print("=" * 80)
    
    # Affichage formaté des métriques
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(metrics_df.to_string())
    print("=" * 80)

    # Vérification de l'effet protecteur
    dd_raw = metrics_df.loc["Max Drawdown (%)", "Stratégie Brute (Sans Filtre)"]
    dd_filt = metrics_df.loc["Max Drawdown (%)", "Stratégie Filtrée (+ Wasserstein CB)"]
    fees_raw = metrics_df.loc["Frais Totaux Payés (%)", "Stratégie Brute (Sans Filtre)"]
    fees_filt = metrics_df.loc["Frais Totaux Payés (%)", "Stratégie Filtrée (+ Wasserstein CB)"]

    print("\nANALYSE QUANTITATIVE :")
    print(f" - Réduction du Max Drawdown : {dd_raw:.2f}%  --->  {dd_filt:.2f}%")
    print(f" - Économie de frais payés   : {fees_raw:.2f}%  --->  {fees_filt:.2f}%")
    if dd_filt < dd_raw:
        print(" [VALIDATION] Le coupe-circuit remplit son rôle de protection contre les chutes brutales !")
    print("=" * 80)


if __name__ == "__main__":
    run_stage_3()
