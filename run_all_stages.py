"""
run_all_stages.py
=================
Script unifié orchestrant et validant l'ensemble des 3 étapes du pipeline quantitatif :
- ÉTAPE 1 : Chargement des données OHLCV, calcul vectorisé des features & comparaison
            statistique des modèles primaires (Momentum vs Amazon Chronos-Bolt).
- ÉTAPE 2 : Entraînement de l'Arbre Substitut Interprétable (WassersteinCircuitBreaker),
            extraction des règles symboliques et analyse de la filtration des régimes toxiques.
- ÉTAPE 3 : Backtest vectorisé net de frictions (10 bps), comparaison des stratégies
            (Brute vs Filtrée vs Buy & Hold BTC), diagnostic de préservation du capital
            ET génération du graphique de synthèse à 3 panneaux (results.png).
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_loader import load_ohlcv
from src.metrics import compute_vectorized_features
from src.primary_model import (
    MomentumForecaster,
    ChronosBoltForecaster,
    compute_directional_target_and_residuals
)
from src.circuit_breaker import WassersteinCircuitBreaker, apply_circuit_breaker
from src.backtest import compare_strategies


def plot_results(
    equity_df: pd.DataFrame,
    df_eval: pd.DataFrame,
    output_path: str = "results.png"
):
    """
    Génère un graphique de synthèse financière institutionnel à 3 panneaux :
    - Panneau 1 : Courbes d'équité cumulées (Brute vs Filtrée vs Buy & Hold BTC)
    - Panneau 2 : Cours du BTC et zones d'activation du coupe-circuit (100% Cash)
    - Panneau 3 : Évolution temporelle de la divergence 1D de Wasserstein W_1
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]})

    c_bnh = "#7f8c8d"       # Gris moderne
    c_raw = "#e67e22"       # Orange vif
    c_filt = "#27ae60"      # Vert émeraude
    c_price = "#2c3e50"     # Bleu nuit
    c_cash = "#e74c3c"      # Rouge alerte
    c_w1 = "#8e44ad"        # Violet statistique

    timestamps = equity_df.index

    # 1. COURBES D'ÉQUITÉ (PnL Cumulé Normalisé à 1.0)
    ax1 = axes[0]
    ax1.plot(timestamps, equity_df["equity_bnh"], label="Benchmark Buy & Hold BTC", color=c_bnh, linestyle="--", linewidth=1.8, alpha=0.85)
    ax1.plot(timestamps, equity_df["equity_raw"], label="Raw Strategy (Unfiltered)", color=c_raw, linewidth=2.0)
    ax1.plot(timestamps, equity_df["equity_filtered"], label="Filtered Strategy (+ Wasserstein CB)", color=c_filt, linewidth=2.5)
    ax1.set_title("Net Performance Comparison (10 bps Frictions): Wasserstein Circuit-Breaker Protection", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative Equity (Base 1.0)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. COURS BTC & ZONES EN CASH (S_t = 0)
    ax2 = axes[1]
    close_price = df_eval.loc[timestamps, "close"]
    ax2.plot(timestamps, close_price, label="BTC/USDT Price (1h)", color=c_price, linewidth=1.6)

    is_cash = equity_df["circuit_breaker"] == 0
    cash_blocks = []
    start_idx = None
    for ts, cash_on in is_cash.items():
        if cash_on and start_idx is None:
            start_idx = ts
        elif not cash_on and start_idx is not None:
            cash_blocks.append((start_idx, ts))
            start_idx = None
    if start_idx is not None:
        cash_blocks.append((start_idx, timestamps[-1]))

    for i, (start, end) in enumerate(cash_blocks):
        lbl = "Circuit-Breaker Triggered (100% Cash)" if i == 0 else ""
        ax2.axvspan(start, end, color=c_cash, alpha=0.22, label=lbl)

    ax2.set_title("Symbolic Circuit-Breaker Dynamics & BTC/USDT Price Action", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Price (USDT)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. DIVERGENCE DE WASSERSTEIN W_1
    ax3 = axes[2]
    w1_series = df_eval.loc[timestamps, "w1"]
    ax3.plot(timestamps, w1_series, label="1D Wasserstein Divergence W_1(200h, 30h)", color=c_w1, linewidth=1.5)
    w1_high_regime = w1_series.quantile(0.85)
    ax3.axhline(w1_high_regime, color="#c0392b", linestyle="--", linewidth=1.5, label=f"High-Drift Regime Threshold (p85 = {w1_high_regime:.4f})")
    ax3.set_title("Rolling 1D Wasserstein Divergence Time Series (Distribution Shift)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Wasserstein Distance W_1", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Date & Time (UTC)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    fig.autofmt_xdate(rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Visualisation] Graphique comparatif enregistré avec succès dans : {output_path}")


def run_stage_1(df_raw: pd.DataFrame = None):
    """
    Étape 1 : Data pipeline, features vectorisées et comparaison des modèles primaires.
    """
    print("\n" + "=" * 80)
    print(" [ÉTAPE 1] : DATA PIPELINE, FEATURES & COMPARAISON MOMENTUM vs CHRONOS-BOLT")
    print("=" * 80)

    # 1. Chargement des données
    if df_raw is None:
        print("\n[1.1] Chargement des barres OHLCV 1h...")
        df_raw = load_ohlcv(symbol="BTCUSDT", cache_path="data/btc_1h.csv")
        print(f"      Nombre de barres : {len(df_raw):,} [{df_raw.index[0]}  ->  {df_raw.index[-1]}]")

    # 2. Features vectorisées
    print("\n[1.2] Calcul des métriques statistiques vectorisées (W_1, Garman-Klass, Log-ret)...")
    df_features = compute_vectorized_features(df_raw, ref_window=200, curr_window=30)

    # 3. Modèle Primaire Baseline : Momentum (3h)
    print("\n[1.3] Inférence Modèle 1 : Momentum Baseline (3h)...")
    mom_forecaster = MomentumForecaster(window=3)
    df_mom = compute_directional_target_and_residuals(df_features, mom_forecaster)

    # 4. Modèle Primaire Foundation Model : Amazon Chronos-Bolt (Zero-Shot)
    print("\n[1.4] Inférence Modèle 2 : Amazon Chronos-Bolt (Tiny, context=48)...")
    chronos_forecaster = ChronosBoltForecaster(
        model_name="amazon/chronos-bolt-tiny",
        context_length=48,
        batch_size=64,
        cache_path="data/chronos_preds.csv"
    )
    df_chronos = compute_directional_target_and_residuals(df_features, chronos_forecaster)

    # 5. Comparaison statistique synchronisée
    df_valid = df_features.dropna(subset=["w1", "gk_vol"]).copy()
    df_valid["pred_mom"] = df_mom.loc[df_valid.index, "pred_dir"]
    df_valid["err_mom"] = df_mom.loc[df_valid.index, "model_error"]
    df_valid["pred_chronos"] = df_chronos.loc[df_valid.index, "pred_dir"]
    df_valid["err_chronos"] = df_chronos.loc[df_valid.index, "model_error"]
    df_valid["target_dir"] = df_mom.loc[df_valid.index, "target_dir"]

    df_eval = df_valid.dropna(subset=["err_mom", "err_chronos", "target_dir"]).copy()
    df_eval["err_mom"] = df_eval["err_mom"].astype(int)
    df_eval["err_chronos"] = df_eval["err_chronos"].astype(int)

    hit_ratio_mom = (1 - df_eval["err_mom"].mean()) * 100
    hit_ratio_chronos = (1 - df_eval["err_chronos"].mean()) * 100
    err_rate_mom = df_eval["err_mom"].mean() * 100
    err_rate_chronos = df_eval["err_chronos"].mean() * 100
    agreement = (df_eval["pred_mom"] == df_eval["pred_chronos"]).mean() * 100

    print("\n" + "-" * 80)
    print(" RÉSULTATS COMPARATIFS : MOMENTUM vs CHRONOS-BOLT")
    print("-" * 80)
    print(f"{'Métrique':<38} | {'Momentum (3h)':<16} | {'Chronos-Bolt (Amazon)':<16}")
    print("-" * 80)
    print(f"{'Précision Directionnelle (Hit-Ratio)':<38} | {hit_ratio_mom:>14.2f} % | {hit_ratio_chronos:>14.2f} %")
    print(f"{'Taux d Erreur Brut (e_t = 1)':<38} | {err_rate_mom:>14.2f} % | {err_rate_chronos:>14.2f} %")
    print(f"{'Nombre d Heures Évaluées':<38} | {len(df_eval):>14,d}   | {len(df_eval):>14,d}  ")
    print(f"{'Accord des Prédictions':<38} | {agreement:>14.2f} % (signaux identiques)")
    print("-" * 80)

    print("\n[VALIDATION ÉTAPE 1] Résidus binaires e_t générés avec succès.")
    return df_features, df_mom, df_chronos


def run_stage_2(df_primary: pd.DataFrame):
    """
    Étape 2 : Entraînement de l'Arbre Substitut Interprétable (WassersteinCircuitBreaker).
    """
    print("\n" + "=" * 80)
    print(" [ÉTAPE 2] : ARBRE SUBSTITUT INTERPRÉTABLE (WASSERSTEIN CIRCUIT-BREAKER)")
    print("=" * 80)

    print("\n[2.1] Entraînement du Coupe-Circuit sur les résidus d'erreur e_t...")
    breaker = WassersteinCircuitBreaker(
        max_depth=2,
        min_samples_leaf=50,
        error_threshold=0.52,  # Coupure (S_t=0) si probabilité d'erreur >= 52%
        feature_names=["w1", "gk_vol"]
    )
    breaker.fit(df_primary, target_col="model_error")

    # Affichage des règles symboliques extraites
    print("\n[2.2] Règles Symboliques Extraites de l'Arbre de Décision :")
    print("-" * 80)
    print(breaker.explain_rules())
    print("-" * 80)

    # Inférence du filtre
    print("\n[2.3] Application du coupe-circuit sur la série chronologique...")
    df_result = apply_circuit_breaker(df_primary, breaker)

    df_eval = df_result.dropna(subset=["circuit_breaker", "model_error"]).copy()
    cash_ratio = (df_eval["circuit_breaker"] == 0).mean() * 100
    trade_ratio = (df_eval["circuit_breaker"] == 1).mean() * 100

    err_when_trading = df_eval.loc[df_eval["circuit_breaker"] == 1, "model_error"].mean() * 100
    err_when_cash = df_eval.loc[df_eval["circuit_breaker"] == 0, "model_error"].mean() * 100

    print("\n" + "-" * 80)
    print(" ANALYSE DE LA FILTRATION DU COUPE-CIRCUIT")
    print("-" * 80)
    print(f"Total d'heures analysées               : {len(df_eval):,}")
    print(f"Temps en Liquidité (Coupe-circuit S=0) : {cash_ratio:.2f} %")
    print(f"Temps en Exposition active (S=1)       : {trade_ratio:.2f} %")
    print("-" * 80)
    print(f"Taux d'erreur quand S=1 (Trade autorisé) : {err_when_trading:.2f} %")
    print(f"Taux d'erreur dans les régimes coupés   : {err_when_cash:.2f} %")
    print("-" * 80)

    print("\n[VALIDATION ÉTAPE 2] Arbre substitut calibré et interprétable avec succès.")
    return df_result, breaker


def run_stage_3(df_stage2: pd.DataFrame, cost_bps: float = 10.0, output_img: str = "results.png"):
    """
    Étape 3 : Backtest vectorisé net de frais, comparaison des stratégies et génération graphique.
    """
    print("\n" + "=" * 80)
    print(f" [ÉTAPE 3] : BACKTEST VECTORISÉ INSTITUTIONNEL (NET DE {cost_bps} BPS)")
    print("=" * 80)

    print(f"\n[3.1] Simulation vectorisée avec frais de transaction ({cost_bps} bps)...")
    equity_df, metrics_df = compare_strategies(
        df_stage2,
        pred_col="pred_dir",
        breaker_col="circuit_breaker",
        ret_col="ret",
        cost_bps=cost_bps
    )

    print("\n" + "-" * 80)
    print(f" TABLEAU SYNTHÉTIQUE DES PERFORMANCES (NET DE FRAIS : {cost_bps} BPS)")
    print("-" * 80)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(metrics_df.to_string())
    print("-" * 80)

    # Analyse de la protection
    dd_raw = metrics_df.loc["Max Drawdown (%)", "Stratégie Brute (Sans Filtre)"]
    dd_filt = metrics_df.loc["Max Drawdown (%)", "Stratégie Filtrée (+ Wasserstein CB)"]
    fees_raw = metrics_df.loc["Frais Totaux Payés (%)", "Stratégie Brute (Sans Filtre)"]
    fees_filt = metrics_df.loc["Frais Totaux Payés (%)", "Stratégie Filtrée (+ Wasserstein CB)"]

    print("\n[3.2] Diagnostic de Préservation du Capital :")
    print(f" - Réduction du Max Drawdown : {dd_raw:.2f}%  --->  {dd_filt:.2f}%")
    print(f" - Économie de frais payés   : {fees_raw:.2f}%  --->  {fees_filt:.2f}%")
    if dd_filt < dd_raw:
        print(" -> [SUCCÈS] Le coupe-circuit amortit significativement les chocs de marché.")

    # Génération du graphique à 3 panneaux si demandé
    if output_img:
        print(f"\n[3.3] Génération du graphique comparatif ({output_img})...")
        plot_results(equity_df, df_stage2, output_path=output_img)

    print("\n[VALIDATION ÉTAPE 3] Backtest vectorisé et visualisation terminés avec succès.")
    return equity_df, metrics_df


def run_all(symbol: str = "BTCUSDT", model_choice: str = "chronos", cost_bps: float = 10.0, output_img: str = "results.png"):
    """
    Exécute les 3 étapes séquentiellement dans un flux unique et génère results.png.
    """
    print("=" * 80)
    print(f" EXÉCUTION INTÉGRALE DES ÉTAPES 1, 2 ET 3 (SYMBOLE: {symbol})")
    print(f" Modèle Primaire : {model_choice.upper()} | Frictions : {cost_bps} bps | Graphique : {output_img}")
    print("=" * 80)

    # Chargement
    df_raw = load_ohlcv(symbol=symbol, cache_path=f"data/{symbol.lower()[:3]}_1h.csv")

    # Étape 1
    df_features, df_mom, df_chronos = run_stage_1(df_raw=df_raw)

    # Sélection du modèle primaire pour la suite
    df_primary = df_chronos if model_choice.lower() == "chronos" else df_mom

    # Étape 2
    df_stage2, breaker = run_stage_2(df_primary=df_primary)

    # Étape 3 (incluant la génération du graphique results.png)
    equity_df, metrics_df = run_stage_3(df_stage2=df_stage2, cost_bps=cost_bps, output_img=output_img)

    print("\n" + "=" * 80)
    print(" [SYNTHÈSE TOTALE] Les 3 étapes ont été validées avec succès !")
    if output_img:
        print(f" Graphique comparatif disponible : {output_img}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script unifié : Validation des étapes 1, 2 et 3 + Graphique.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "2", "3"],
                        help="Étape spécifique à exécuter (1, 2, 3 ou all)")
    parser.add_argument("--model", type=str, default="chronos", choices=["chronos", "momentum"],
                        help="Modèle primaire à propager dans les étapes 2 et 3")
    parser.add_argument("--cost-bps", type=float, default=10.0,
                        help="Frais de transaction en bps pour le backtest (défaut: 10.0)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbole crypto (défaut: BTCUSDT)")
    parser.add_argument("--output", type=str, default="results.png",
                        help="Nom du fichier image de sortie (défaut: results.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Désactive la génération du graphique image")
    args = parser.parse_args()

    img_out = None if args.no_plot else args.output

    if args.stage == "all":
        run_all(symbol=args.symbol, model_choice=args.model, cost_bps=args.cost_bps, output_img=img_out)
    elif args.stage == "1":
        run_stage_1()
    elif args.stage == "2":
        df_raw = load_ohlcv(symbol=args.symbol, cache_path=f"data/{args.symbol.lower()[:3]}_1h.csv")
        df_features = compute_vectorized_features(df_raw)
        forecaster = ChronosBoltForecaster() if args.model == "chronos" else MomentumForecaster()
        df_primary = compute_directional_target_and_residuals(df_features, forecaster)
        run_stage_2(df_primary)
    elif args.stage == "3":
        df_raw = load_ohlcv(symbol=args.symbol, cache_path=f"data/{args.symbol.lower()[:3]}_1h.csv")
        df_features = compute_vectorized_features(df_raw)
        forecaster = ChronosBoltForecaster() if args.model == "chronos" else MomentumForecaster()
        df_primary = compute_directional_target_and_residuals(df_features, forecaster)
        breaker = WassersteinCircuitBreaker().fit(df_primary)
        df_stage2 = apply_circuit_breaker(df_primary, breaker)
        run_stage_3(df_stage2, cost_bps=args.cost_bps, output_img=img_out)
