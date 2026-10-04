"""
backtest.py
===========
Module de l'Étape 3 : Moteur de Backtest Vectorisé Net de Frais
Permet de simuler et comparer les performances financières des stratégies
brutes, filtrées par coupe-circuit (Circuit-Breaker) et du benchmark Buy & Hold.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def backtest_single_strategy(
    returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 10.0,
    is_log_return: bool = True
) -> pd.DataFrame:
    """
    Simule une stratégie de trading vectorisée avec déduction stricte des frais.

    Règle temporelle (No Lookahead Bias) :
    - La position pos_{t-1} prise à la clôture de la barre précédente s'applique
      au rendement r_t de la barre actuelle : r_gross,t = pos_{t-1} * r_t
    - Les frais de transaction sont prélevés lors du changement de position :
      Delta pos_t = |pos_t - pos_{t-1}|
      fee_t = (cost_bps / 10,000) * Delta pos_t

    Paramètres
    ----------
    returns : pd.Series
        Série temporelle des rendements de l'actif sous-jacent (ex. log-rendements r_t).
    positions : pd.Series
        Série temporelle de l'exposition voulue : pos_t in {-1, 0, +1}.
    cost_bps : float
        Frais de transaction en points de base (ex. 10.0 bps = 0.10%).
    is_log_return : bool
        Si True, cumule les rendements via exponentielle de la somme.
        Si False, cumule via produit (1 + r).

    Retourne
    --------
    pd.DataFrame contenant :
    - 'position' : position effective à chaque instant t
    - 'turnover' : variation absolue de position |pos_t - pos_{t-1}|
    - 'fee' : coût de transaction prélevé à t
    - 'ret_gross' : rendement brut à t
    - 'ret_net' : rendement net de frais à t
    - 'equity' : courbe de capital normalisée (départ à 1.0)
    - 'drawdown' : perte relative depuis le pic historique
    """
    # Alignement temporel strict
    df = pd.DataFrame({"ret": returns, "pos": positions}).dropna()

    pos = df["pos"]
    ret = df["ret"]

    # 1. Turnover & Frais de transaction
    # À t=0, on suppose un passage de Cash (0) à pos[0]
    pos_shifted = pos.shift(1).fillna(0.0)
    turnover = (pos - pos_shifted).abs()
    fee_rate = cost_bps / 10_000.0
    fee = turnover * fee_rate

    # 2. Rendement Brut et Net (pos_{t-1} * r_t)
    ret_gross = pos_shifted * ret
    ret_net = ret_gross - fee

    # 3. Courbe d'Équité (Capital normalisé à 1.0)
    if is_log_return:
        equity = np.exp(ret_net.cumsum())
    else:
        equity = (1.0 + ret_net).cumprod()

    # 4. Drawdown historique
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max

    out = pd.DataFrame(
        {
            "position": pos,
            "turnover": turnover,
            "fee": fee,
            "ret_gross": ret_gross,
            "ret_net": ret_net,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=df.index
    )

    return out


def compute_performance_metrics(
    bt_result: pd.DataFrame,
    periods_per_year: int = 8760
) -> Dict[str, float]:
    """
    Calcule les métriques financières clés d'une série de backtest.

    Paramètres
    ----------
    bt_result : pd.DataFrame
        Sortie de backtest_single_strategy contenant 'ret_net', 'equity', 'position', 'fee'.
    periods_per_year : int
        Nombre de périodes par an (défaut = 8760 pour des barres 1 heure en crypto 24/7).

    Retourne
    --------
    Dict[str, float] contenant :
    - Rendement Total Net (%)
    - Rendement Annualisé (%)
    - Volatilité Annualisée (%)
    - Sharpe Ratio Annualisé
    - Maximum Drawdown (%)
    - Ratio de Calmar
    - Taux en Cash (%)
    - Frais Totaux Payés (%)
    - Nombre de Trades (Rotations)
    - Win Rate sur trades (%)
    """
    ret_net = bt_result["ret_net"]
    equity = bt_result["equity"]
    pos = bt_result["position"]
    fees = bt_result["fee"]
    drawdown = bt_result["drawdown"]

    n_bars = len(ret_net)
    if n_bars == 0:
        return {}

    # Rendement Total et Annualisé
    total_ret = float(equity.iloc[-1] - 1.0)
    n_years = n_bars / float(periods_per_year)
    if n_years > 0 and equity.iloc[-1] > 0:
        cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    else:
        cagr = 0.0

    # Volatilité et Sharpe Ratio
    std_ret = float(ret_net.std())
    mean_ret = float(ret_net.mean())
    ann_vol = std_ret * np.sqrt(periods_per_year)
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    # Maximum Drawdown & Calmar
    max_dd = float(drawdown.max())
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    # Exposition
    cash_ratio = float((pos == 0).mean())
    total_fees = float(fees.sum())
    n_trades = int((bt_result["turnover"] > 0).sum())

    # Win Rate (barres investies où ret_net > 0)
    active_bars = ret_net[pos.shift(1) != 0]
    win_rate = float((active_bars > 0).mean()) if len(active_bars) > 0 else 0.0

    return {
        "Total Return Net (%)": total_ret * 100.0,
        "CAGR Annualisé (%)": cagr * 100.0,
        "Volatilité Ann. (%)": ann_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100.0,
        "Calmar Ratio": calmar,
        "Temps en Cash (%)": cash_ratio * 100.0,
        "Frais Totaux Payés (%)": total_fees * 100.0,
        "Nombre de Changements Pos": n_trades,
        "Win Rate Barres (%)": win_rate * 100.0,
    }


def compare_strategies(
    df: pd.DataFrame,
    pred_col: str = "pred_dir",
    breaker_col: str = "circuit_breaker",
    ret_col: str = "ret",
    cost_bps: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute simultanément le backtest des 3 stratégies :
    1. Stratégie Brute (sans filtre : pos = pred_dir)
    2. Stratégie Filtrée (+ Coupe-circuit : pos = pred_dir * circuit_breaker)
    3. Benchmark Buy & Hold (pos = 1)

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame enrichi contenant les rendements, les prédictions et le signal du circuit-breaker.
    pred_col : str
        Nom de la colonne des prédictions directionnelles primaires {-1, +1}.
    breaker_col : str
        Nom de la colonne du signal de coupe-circuit {0, 1}.
    ret_col : str
        Nom de la colonne des rendements de marché (ex. 'ret').
    cost_bps : float
        Frais de transaction en bps.

    Retourne
    --------
    equity_df : pd.DataFrame
        DataFrame contenant l'historique des courbes d'équité comparées.
    metrics_df : pd.DataFrame
        Tableau récapitulatif comparatif des métriques financières.
    """
    valid_cols = [ret_col, pred_col, breaker_col]
    df_clean = df.dropna(subset=valid_cols).copy()

    # 1. Définition des positions
    pos_raw = df_clean[pred_col]
    pos_filtered = df_clean[pred_col] * df_clean[breaker_col]
    pos_bnh = pd.Series(1.0, index=df_clean.index)

    # 2. Exécution des backtests vectorisés
    bt_raw = backtest_single_strategy(df_clean[ret_col], pos_raw, cost_bps=cost_bps)
    bt_filt = backtest_single_strategy(df_clean[ret_col], pos_filtered, cost_bps=cost_bps)
    bt_bnh = backtest_single_strategy(df_clean[ret_col], pos_bnh, cost_bps=0.0)  # Buy & Hold pas de turnover

    # 3. Assemblage des courbes d'équité
    equity_df = pd.DataFrame({
        "close": df_clean["close"] if "close" in df_clean.columns else np.nan,
        "equity_raw": bt_raw["equity"],
        "equity_filtered": bt_filt["equity"],
        "equity_bnh": bt_bnh["equity"],
        "drawdown_raw": bt_raw["drawdown"],
        "drawdown_filtered": bt_filt["drawdown"],
        "pos_filtered": pos_filtered,
        "circuit_breaker": df_clean[breaker_col],
    }, index=df_clean.index)

    # 4. Calcul des métriques pour chaque stratégie
    metrics_raw = compute_performance_metrics(bt_raw)
    metrics_filt = compute_performance_metrics(bt_filt)
    metrics_bnh = compute_performance_metrics(bt_bnh)

    metrics_df = pd.DataFrame({
        "Stratégie Brute (Sans Filtre)": metrics_raw,
        "Stratégie Filtrée (+ Wasserstein CB)": metrics_filt,
        "Benchmark Buy & Hold": metrics_bnh,
    })

    return equity_df, metrics_df
