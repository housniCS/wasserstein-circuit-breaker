"""
run_pipeline.py
===============
Pipeline unifié de bout en bout :
1. Extraction / Chargement des données OHLCV 1h BTC/USDT.
2. Extraction vectorisée des métriques de régime (Garman-Klass, Wasserstein W_1).
3. Modèle primaire directionnel (Amazon Chronos-Bolt Foundation Model / Momentum).
4. Entraînement de l'Arbre Substitut Interprétable (WassersteinCircuitBreaker).
5. Backtest vectorisé net de frais (10 bps).
6. Génération de la visualisation comparative en 3 panneaux (results.png).
"""

import os
import argparse
import numpy as np
import pandas as pd
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
from src.backtest import compare_strategies, compute_performance_metrics


def plot_results(
    equity_df: pd.DataFrame,
    df_eval: pd.DataFrame,
    breaker: WassersteinCircuitBreaker,
    output_path: str = "results.png"
):
    """
    Génère un graphique de synthèse financière à 3 panneaux :
    - Panneau 1 : Courbes d'équité cumulées (Brute vs Filtrée vs Buy & Hold)
    - Panneau 2 : Cours du BTC et zones d'activation du coupe-circuit (100% Cash)
    - Panneau 3 : Évolution de la divergence 1D de Wasserstein W_1 et seuil critique
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]})

    # Palette de couleurs institutionnelle
    c_bnh = "#7f8c8d"       # Gris moderne
    c_raw = "#e67e22"       # Orange vif
    c_filt = "#27ae60"      # Vert émeraude
    c_price = "#2c3e50"     # Bleu nuit
    c_cash = "#e74c3c"      # Rouge alerte
    c_w1 = "#8e44ad"        # Violet statistique

    timestamps = equity_df.index

    # -------------------------------------------------------------
    # PANNEAU 1 : COURBES D'ÉQUITÉ (PnL Cumulé Normalisé à 1.0)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(timestamps, equity_df["equity_bnh"], label="Benchmark Buy & Hold BTC", color=c_bnh, linestyle="--", linewidth=1.8, alpha=0.85)
    ax1.plot(timestamps, equity_df["equity_raw"], label="Raw Strategy (Chronos-Bolt)", color=c_raw, linewidth=2.0)
    ax1.plot(timestamps, equity_df["equity_filtered"], label="Filtered Strategy (+ Wasserstein CB)", color=c_filt, linewidth=2.5)

    ax1.set_title("Net Performance Comparison (10 bps Frictions): Wasserstein Circuit-Breaker Protection", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative Equity (Base 1.0)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # PANNEAU 2 : DYNAMIQUE DU COUPE-CIRCUIT & COURS DU BTC
    # -------------------------------------------------------------
    ax2 = axes[1]
    close_price = df_eval.loc[timestamps, "close"]
    ax2.plot(timestamps, close_price, label="BTC/USDT Price (1h)", color=c_price, linewidth=1.6)

    # Surlignage rouge des périodes où le coupe-circuit force le Cash (S_t = 0)
    is_cash = equity_df["circuit_breaker"] == 0
    cash_blocks = []
    start_idx = None

    for i, (ts, cash_on) in enumerate(is_cash.items()):
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

    # -------------------------------------------------------------
    # PANNEAU 3 : ÉVOLUTION DE LA DISTANCE DE WASSERSTEIN W_1
    # -------------------------------------------------------------
    ax3 = axes[2]
    w1_series = df_eval.loc[timestamps, "w1"]
    ax3.plot(timestamps, w1_series, label="1D Wasserstein Divergence W_1(200h, 30h)", color=c_w1, linewidth=1.5)

    # Ligne de seuil de référence (médiane supérieure des régimes de rupture)
    w1_high_regime = w1_series.quantile(0.85)
    ax3.axhline(w1_high_regime, color="#c0392b", linestyle="--", linewidth=1.5, label=f"High-Drift Regime Threshold (p85 = {w1_high_regime:.4f})")

    ax3.set_title("Rolling 1D Wasserstein Divergence Time Series (Distribution Shift)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Wasserstein Distance W_1", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Date & Time (UTC)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Format de l'axe des abscisses (dates)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %Hh"))
    fig.autofmt_xdate(rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Visualisation] Graphique comparatif enregistré avec succès dans : {output_path}")


def run_full_pipeline(
    symbol: str = "BTCUSDT",
    cost_bps: float = 10.0,
    model_choice: str = "chronos",
    output_img: str = "results.png"
):
    print("=" * 80)
    print(" PIPELINE UNIFIÉ : DÉTECTION DE DISTRIBUTION-SHIFT & COUPE-CIRCUIT")
    print(f" Actif: {symbol} | Frais: {cost_bps} bps | Modèle Primaire: {model_choice.upper()}")
    print("=" * 80)

    # 1. Pipeline de Données
    print("\n[Étape 1/5] Chargement des données OHLCV 1h...")
    df_raw = load_ohlcv(symbol=symbol, cache_path="data/btc_1h.csv")
    print(f"            {len(df_raw)} barres chargées [{df_raw.index[0]}  ->  {df_raw.index[-1]}]")

    # 2. Métriques Vectorisées
    print("\n[Étape 2/5] Calcul vectorisé des métriques de régime (Garman-Klass & Wasserstein W_1)...")
    df_features = compute_vectorized_features(df_raw, ref_window=200, curr_window=30)

    # 3. Modèle Primaire & Cible d'Erreur e_t
    print(f"\n[Étape 3/5] Inférence du modèle primaire ({model_choice.upper()}) & Résidu d'erreur e_t...")
    if model_choice.lower() == "chronos":
        forecaster = ChronosBoltForecaster(
            model_name="amazon/chronos-bolt-tiny",
            context_length=48,
            batch_size=64,
            cache_path="data/chronos_preds.csv"
        )
    else:
        forecaster = MomentumForecaster(window=3)

    df_stage1 = compute_directional_target_and_residuals(df_features, forecaster)

    # 4. Entraînement du Coupe-Circuit Symbolique (Wasserstein Surrogate Tree)
    print("\n[Étape 4/5] Entraînement de l'Arbre Substitut Interprétable (WassersteinCircuitBreaker)...")
    breaker = WassersteinCircuitBreaker(
        max_depth=2,
        min_samples_leaf=50,
        error_threshold=0.52,
        feature_names=["w1", "gk_vol"]
    )
    breaker.fit(df_stage1, target_col="model_error")
    df_stage2 = apply_circuit_breaker(df_stage1, breaker)

    # Affichage des règles symboliques
    print(breaker.explain_rules())

    # 5. Backtest Vectorisé Net de Frais
    print(f"\n[Étape 5/5] Exécution du backtest vectorisé net de frais ({cost_bps} bps)...")
    equity_df, metrics_df = compare_strategies(
        df_stage2,
        pred_col="pred_dir",
        breaker_col="circuit_breaker",
        ret_col="ret",
        cost_bps=cost_bps
    )

    # -------------------------------------------------------------
    # SYNTHÈSE DES RÉSULTATS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f" TABLEAU SYNTHÉTIQUE DES PERFORMANCES COMPARÉES (NET DE {cost_bps} BPS)")
    print("=" * 80)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(metrics_df.to_string())
    print("=" * 80)

    # Génération du graphique 3 panneaux
    plot_results(equity_df, df_stage2, breaker, output_path=output_img)

    print("\n" + "=" * 80)
    print(" [CONFIRMATION] Pipeline complet exécuté avec succès.")
    print(f" Rapport graphique disponible : {output_img}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet : Distribution-Shift & Circuit-Breaker")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole crypto (défaut: BTCUSDT)")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Frais de transaction en bps (défaut: 10.0)")
    parser.add_argument("--model", type=str, default="chronos", choices=["chronos", "momentum"], help="Modèle primaire (chronos ou momentum)")
    parser.add_argument("--output", type=str, default="results.png", help="Chemin de sauvegarde du graphique")
    args = parser.parse_args()

    run_full_pipeline(
        symbol=args.symbol,
        cost_bps=args.cost_bps,
        model_choice=args.model,
        output_img=args.output
    )
