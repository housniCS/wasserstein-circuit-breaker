"""
Wasserstein Circuit-Breaker Package
===================================
Distribution-shift resilience and model risk management for algorithmic trading.

Modules:
    - data_loader: Fetching and synthetic generation of OHLCV data.
    - metrics: Vectorized computation of Garman-Klass volatility and rolling 1D Wasserstein distance.
    - primary_model: Momentum and Amazon Chronos-Bolt forecasting models, residual calculation.
    - circuit_breaker: Interpretable CART surrogate decision tree for risk-off cash filtering.
    - backtest: Vectorized net-of-fees backtesting and financial performance metrics.
"""

from src.data_loader import load_ohlcv, fetch_binance_ohlcv, generate_synthetic_ohlcv
from src.metrics import compute_garman_klass_volatility, compute_rolling_wasserstein_1d, compute_vectorized_features
from src.primary_model import MomentumForecaster, ChronosBoltForecaster, compute_directional_target_and_residuals
from src.circuit_breaker import WassersteinCircuitBreaker, apply_circuit_breaker
from src.backtest import backtest_single_strategy, compare_strategies, compute_performance_metrics

__all__ = [
    "load_ohlcv",
    "fetch_binance_ohlcv",
    "generate_synthetic_ohlcv",
    "compute_garman_klass_volatility",
    "compute_rolling_wasserstein_1d",
    "compute_vectorized_features",
    "MomentumForecaster",
    "ChronosBoltForecaster",
    "compute_directional_target_and_residuals",
    "WassersteinCircuitBreaker",
    "apply_circuit_breaker",
    "backtest_single_strategy",
    "compare_strategies",
    "compute_performance_metrics",
]
