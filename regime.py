"""Regime detection. Stable, not noisy. Persistence filter prevents flips."""

import numpy as np
import pandas as pd
from config import REGIME_ATR_PERIOD, REGIME_VOLATILE_MULT

# New params — intentionally NOT in config yet, validated here first
REGIME_EMA_PERIOD = 100      # 100 hours = ~4 days (was 20 = too noisy)
REGIME_PERSISTENCE = 12      # 12 bars = 12 hours before regime switch
REGIME_TREND_THRESHOLD = 0.02  # 2% from EMA (was 1% = too tight)


def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_regime(df):
    """
    Stable regime detection. No rapid flipping.
    Uses EMA(100), 2% trend threshold, 12-bar persistence filter.
    Returns: 'trending_up', 'trending_down', 'ranging', 'volatile'
    """
    df = df.copy()
    df["ema"] = df["close"].ewm(span=REGIME_EMA_PERIOD, adjust=False).mean()
    df["atr"] = calc_atr(df, REGIME_ATR_PERIOD)
    df["atr_avg"] = df["atr"].rolling(50).mean()  # Long-term ATR average

    # Raw regime: what does price say RIGHT NOW?
    raw_regimes = []
    for i in range(len(df)):
        if i < REGIME_EMA_PERIOD:
            raw_regimes.append("unknown")
            continue

        close = df["close"].iloc[i]
        ema = df["ema"].iloc[i]
        atr = df["atr"].iloc[i]
        atr_avg = df["atr_avg"].iloc[i]

        if atr_avg > 0 and atr > atr_avg * REGIME_VOLATILE_MULT:
            raw_regimes.append("volatile")
        else:
            pct = (close - ema) / ema if ema > 0 else 0
            if abs(pct) > REGIME_TREND_THRESHOLD:
                raw_regimes.append("trending_up" if pct > 0 else "trending_down")
            else:
                raw_regimes.append("ranging")

    # Persistence filter: regime only changes after N consecutive bars
    # of the SAME raw regime signal. Until then, keep previous regime.
    regimes = []
    current_regime = "unknown"
    streak_regime = None
    streak_count = 0

    for i, raw in enumerate(raw_regimes):
        if raw == current_regime or streak_count == 0:
            # Keep current, reset streak
            streak_regime = raw
            streak_count = 1
            regimes.append(current_regime if i >= REGIME_EMA_PERIOD else raw)
        elif raw == streak_regime:
            # Same as what we're tracking, increment
            streak_count += 1
            if streak_count >= REGIME_PERSISTENCE:
                # Confirmed — switch regime
                current_regime = raw
                streak_count = 0
            regimes.append(current_regime)
        else:
            # Different regime — reset streak tracking
            streak_regime = raw
            streak_count = 1
            regimes.append(current_regime)

    df["regime"] = regimes
    return df