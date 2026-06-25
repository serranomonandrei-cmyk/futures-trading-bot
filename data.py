"""Fetch OHLCV from Binance. Minimal, no abstraction."""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone


def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def fetch_ohlcv(exchange, symbol, timeframe="1h", months=6):
    """Fetch historical OHLCV. Returns DataFrame with UTC timestamps."""
    since = int((datetime.now(timezone.utc) - timedelta(days=months * 30)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def resample_ohlcv(df_1h, target_tf):
    """Resample 1H data to target timeframe."""
    if target_tf == "1h":
        return df_1h.copy()

    tf_map = {"5m": "5min", "15m": "15min", "30m": "30min", "4h": "4h", "1d": "1D"}
    rule = tf_map.get(target_tf)
    if not rule:
        raise ValueError(f"Unknown timeframe: {target_tf}")

    df = df_1h.set_index("timestamp").copy()
    resampled = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return resampled
