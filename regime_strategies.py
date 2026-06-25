"""
Regime-specific strategies. Designed for each market condition.
Tested on hardcoded regime periods.
"""

import numpy as np
import pandas as pd


# === INDICATOR HELPERS ===

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = -d.clip(upper=0).rolling(p).mean()
    return 100 - 100 / (1 + up / dn)

def donchian(df, period=20):
    return df["high"].rolling(period).max(), df["low"].rolling(period).min()

def bollinger(s, period=20, std_mult=2.0):
    mid = sma(s, period)
    std = s.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std


# ================================================================
# BULL STRATEGY: Buy dips in uptrend
# ================================================================
# Logic: Price is above 50 EMA (uptrend), pulls back to 20 EMA,
# RSI oversold on pullback, volume confirms bounce.
# Entry: Close above 20 EMA after pullback.
# Stop: Below recent swing low (1.5x ATR from entry)
# TP: 2x risk

def bull_pullback(df, ema_fast=20, ema_slow=50, rsi_period=14, rsi_os=35):
    """
    Buy dips in uptrend:
    1. Close > 50 EMA (uptrend)
    2. RSI < rsi_os (pullback)
    3. Close > 20 EMA (bounce confirmation)
    4. Previous bar: close was near/below 20 EMA
    """
    e_fast = ema(df["close"], ema_fast)
    e_slow = ema(df["close"], ema_slow)
    r = rsi(df["close"], rsi_period)

    # Pullback condition: price was near 20 EMA
    near_ema = (df["low"] <= e_fast * 1.005) | (df["close"].shift(1) <= e_fast.shift(1) * 1.005)

    sig = pd.Series(0, index=df.index)
    long = (
        (df["close"] > e_slow) &        # Uptrend
        (r < rsi_os) &                   # Oversold pullback
        (df["close"] > e_fast) &         # Bounced above fast EMA
        near_ema                         # Was near EMA
    )
    # Only on first signal (crossover from non-long to long)
    sig[long & ~long.shift(1).fillna(False)] = 1
    return sig


# ================================================================
# BEAR STRATEGY: Short rallies in downtrend
# ================================================================
# Logic: Price is below 50 EMA (downtrend), rallies to 20 EMA,
# RSI overbought on rally, volume confirms rejection.
# Entry: Close below 20 EMA after rally.
# Stop: Above recent swing high (1.5x ATR from entry)
# TP: 2x risk

def bear_pullback(df, ema_fast=20, ema_slow=50, rsi_period=14, rsi_ob=65):
    """
    Short rallies in downtrend:
    1. Close < 50 EMA (downtrend)
    2. RSI > rsi_ob (overbought rally)
    3. Close < 20 EMA (rejection confirmation)
    4. Previous bar: price was near/above 20 EMA
    """
    e_fast = ema(df["close"], ema_fast)
    e_slow = ema(df["close"], ema_slow)
    r = rsi(df["close"], rsi_period)

    near_ema = (df["high"] >= e_fast * 0.995) | (df["close"].shift(1) >= e_fast.shift(1) * 0.995)

    sig = pd.Series(0, index=df.index)
    short = (
        (df["close"] < e_slow) &        # Downtrend
        (r > rsi_ob) &                   # Overbought rally
        (df["close"] < e_fast) &         # Rejected below fast EMA
        near_ema                         # Was near EMA
    )
    sig[short & ~short.shift(1).fillna(False)] = -1
    return sig


# ================================================================
# RANGING STRATEGY: Fade extremes at range boundaries
# ================================================================
# Logic: Price at Bollinger Band extreme, RSI confirms extreme,
# fade back to middle band. Small target, tight stop.

def range_fade(df, bb_period=20, bb_std=2.0, rsi_period=14, rsi_os=30, rsi_ob=70):
    """
    Fade range extremes:
    1. Price touches/breaks Bollinger Band
    2. RSI confirms extreme
    3. Fade to middle band
    """
    mid, upper, lower = bollinger(df["close"], bb_period, bb_std)
    r = rsi(df["close"], rsi_period)
    a = atr(df, 14)

    sig = pd.Series(0, index=df.index)

    # Long: price at lower band, RSI oversold
    long = (
        (df["close"] <= lower) &
        (r < rsi_os)
    )
    sig[long & ~long.shift(1).fillna(False)] = 1

    # Short: price at upper band, RSI overbought
    short = (
        (df["close"] >= upper) &
        (r > rsi_ob)
    )
    sig[short & ~short.shift(1).fillna(False)] = -1

    return sig


# ================================================================
# RANGING STRATEGY v2: Range breakout (buy support, sell resistance)
# ================================================================

def range_swing(df, lookback=48, rsi_period=14, rsi_os=35, rsi_ob=65):
    """
    Buy at support, sell at resistance:
    1. Identify range from recent high/low
    2. Buy at support (low of range) with RSI confirmation
    3. Sell at resistance (high of range)
    """
    # Dynamic support/resistance from recent bars
    high_range = df["high"].rolling(lookback).max()
    low_range = df["low"].rolling(lookback).min()
    mid_range = (high_range + low_range) / 2
    r = rsi(df["close"], rsi_period)

    # How far price is from range boundaries (0 = at low, 1 = at high)
    range_size = high_range - low_range
    pos_in_range = (df["close"] - low_range) / (range_size + 1e-10)

    sig = pd.Series(0, index=df.index)

    # Buy near support (bottom 15% of range) + oversold
    long = (pos_in_range < 0.15) & (r < rsi_os)
    sig[long & ~long.shift(1).fillna(False)] = 1

    # Short near resistance (top 15% of range) + overbought
    short = (pos_in_range > 0.85) & (r > rsi_ob)
    sig[short & ~short.shift(1).fillna(False)] = -1

    return sig


# ================================================================
# RANGING STRATEGY v3: Mean reversion with Bollinger squeeze
# ================================================================

def range_squeeze(df, bb_period=20, bb_std=1.5, rsi_period=14):
    """
    Bollinger squeeze + fade:
    1. BB width is narrow (squeeze)
    2. Price touches band
    3. Fade to middle
    """
    mid, upper, lower = bollinger(df["close"], bb_period, bb_std)
    bb_width = (upper - lower) / mid
    bb_width_avg = bb_width.rolling(50).mean()
    r = rsi(df["close"], rsi_period)

    sig = pd.Series(0, index=df.index)

    # Squeeze: BB width below average
    is_squeeze = bb_width < bb_width_avg * 0.8

    # Long: squeeze + at lower band + oversold
    long = is_squeeze & (df["close"] <= lower) & (r < 40)
    sig[long & ~long.shift(1).fillna(False)] = 1

    # Short: squeeze + at upper band + overbought
    short = is_squeeze & (df["close"] >= upper) & (r > 60)
    sig[short & ~short.shift(1).fillna(False)] = -1

    return sig


# ================================================================
# REGISTRY
# ================================================================

BULL_STRATEGIES = {
    "bull_pullback": bull_pullback,
}

BEAR_STRATEGIES = {
    "bear_pullback": bear_pullback,
}

RANGE_STRATEGIES = {
    "range_fade": range_fade,
    "range_swing": range_swing,
    "range_squeeze": range_squeeze,
}

ALL_REGIME_STRATEGIES = {**BULL_STRATEGIES, **BEAR_STRATEGIES, **RANGE_STRATEGIES}
