"""
data_loader.py
Module de chargement des données OHLCV 1h pour BTC/USDT.
Supporte :
1. Téléchargement direct depuis l'API publique Binance (sans clé requise).
2. Mise en cache locale en CSV.
3. Fallback synthétique avec sauts de régimes (reproductible).
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import requests


def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """
    Récupère les bougies historiques récentes depuis l'API REST publique de Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw_data = response.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(raw_data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp")
    return df


def generate_synthetic_ohlcv(n_bars: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Génère une série temporelle synthétique OHLCV 1h reproduisant :
    - Un régime standard (faible volatilité gaussienne)
    - Deux régimes de choc (volatilité forte, sauts laplaciens et asymétrie)
    """
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    # Rendements de base avec 2 sauts de régimes
    returns = np.random.normal(0, 0.008, n_bars)
    # Régime 1 : Choc laplacien (queues épaisses)
    returns[1000:1300] = np.random.laplace(0, 0.025, 300)
    # Régime 2 : Tendance baissière et haute volatilité
    returns[2200:2400] = np.random.normal(-0.005, 0.03, 200)

    # Reconstitution des cours
    close = 50000.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.004, n_bars)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.004, n_bars)))
    open_p = close * (1.0 + np.random.normal(0, 0.002, n_bars))
    volume = np.random.lognormal(mean=5.0, sigma=0.5, size=n_bars) * 100

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }).set_index("timestamp")
    return df


def load_ohlcv(
    symbol: str = "BTCUSDT",
    cache_path: str = "data/btc_1h.csv",
    use_synthetic_fallback: bool = True,
    n_synthetic_bars: int = 3000
) -> pd.DataFrame:
    """
    Point d'entrée principal pour charger les données OHLCV.
    Tente de charger le cache local, sinon télécharge via l'API, sinon utilise le générateur synthétique.
    """
    if os.path.exists(cache_path):
        print(f"[Data] Chargement depuis le cache local : {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    # Tentative de téléchargement
    try:
        print(f"[Data] Téléchargement OHLCV 1h pour {symbol} depuis Binance...")
        df = fetch_binance_ohlcv(symbol=symbol, limit=1000)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
        print(f"[Data] Données sauvegardées dans : {cache_path} ({len(df)} barres)")
        return df
    except Exception as e:
        print(f"[Data] Avertissement : impossible de contacter l'API ({e}).")
        if use_synthetic_fallback:
            print(f"[Data] Génération d'une série synthétique réaliste ({n_synthetic_bars} barres)...")
            df = generate_synthetic_ohlcv(n_bars=n_synthetic_bars)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
            return df
        raise
