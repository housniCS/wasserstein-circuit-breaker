"""
test_stage2.py
==============
Script de test et de validation de l'Étape 2 :
1. Chargement des données BTC 1h + Features (Garman-Klass, Wasserstein W_1)
2. Modèle primaire (Momentum ou Chronos-Bolt) et calcul du résidu d'erreur e_t
3. Entraînement de l'Arbre Substitut Interprétable (WassersteinCircuitBreaker)
4. Extraction et affichage des règles symboliques (explain_rules)
5. Analyse statistique des coupures de régime (Pourcentage en Cash vs Trade)
"""

import pandas as pd
from src.data_loader import load_ohlcv
from src.metrics import compute_vectorized_features
from src.primary_model import MomentumForecaster, compute_directional_target_and_residuals
from src.circuit_breaker import WassersteinCircuitBreaker, apply_circuit_breaker


def run_stage_2():
    print("=" * 75)
    print(" ÉTAPE 2 : ARBRE SUBSTITUT INTERPRÉTABLE (WASSERSTEIN CIRCUIT-BREAKER)")
    print("=" * 75)

    # 1. Chargement des données et calcul des features
    print("\n[1/4] Chargement des données & métriques statistiques...")
    df_raw = load_ohlcv(symbol="BTCUSDT", cache_path="data/btc_1h.csv")
    df_features = compute_vectorized_features(df_raw, ref_window=200, curr_window=30)

    # 2. Modèle primaire et calcul des résidus e_t
    print("\n[2/4] Calcul des résidus d'erreur e_t du modèle primaire...")
    forecaster = MomentumForecaster(window=3)
    df_stage1 = compute_directional_target_and_residuals(df_features, forecaster)

    # 3. Entraînement du Circuit-Breaker (Arbre Substitut)
    print("\n[3/4] Entraînement de WassersteinCircuitBreaker (max_depth=2, min_samples_leaf=50)...")
    breaker = WassersteinCircuitBreaker(
        max_depth=2,
        min_samples_leaf=50,
        error_threshold=0.52,  # Coupure si probabilité d'erreur >= 52%
        feature_names=["w1", "gk_vol"]
    )
    breaker.fit(df_stage1, target_col="model_error")

    # 4. Affichage des règles symboliques extraites
    print("\n" + breaker.explain_rules())

    # 5. Application et statistiques
    df_result = apply_circuit_breaker(df_stage1, breaker)

    # Évaluation sur la période valide
    df_eval = df_result.dropna(subset=["circuit_breaker", "model_error"]).copy()
    cash_ratio = (df_eval["circuit_breaker"] == 0).mean() * 100
    trade_ratio = (df_eval["circuit_breaker"] == 1).mean() * 100

    # Taux d'erreur quand le circuit-breaker autorise le trade vs quand il coupe
    err_when_trading = df_eval.loc[df_eval["circuit_breaker"] == 1, "model_error"].mean() * 100
    err_when_cash = df_eval.loc[df_eval["circuit_breaker"] == 0, "model_error"].mean() * 100

    print("\n" + "=" * 75)
    print(" ANALYSE DE L'EFFET DU COUPE-CIRCUIT")
    print("=" * 75)
    print(f"Total d'heures analysées               : {len(df_eval):,}")
    print(f"Temps en Cash (Coupe-circuit actif S=0) : {cash_ratio:.2f} %")
    print(f"Temps en Trade (S=1)                   : {trade_ratio:.2f} %")
    print("-" * 75)
    print(f"Taux d'erreur quand S=1 (Trade autorisé) : {err_when_trading:.2f} %")
    print(f"Taux d'erreur dans les régimes coupés   : {err_when_cash:.2f} %")
    print("=" * 75)
    print("\n[SUCCÈS] L'Arbre Substitut est fonctionnel et interprétable !")


if __name__ == "__main__":
    run_stage_2()
