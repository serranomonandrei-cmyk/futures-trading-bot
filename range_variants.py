"""
Ranging strategy variants. Testing EVERY approach to find what works in ranges.
"""

import numpy as np
import pandas as pd


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = -d.clip(upper=0).rolling(p).mean()
    return 100 - 100 / (1 + up / dn)

def bollinger(s, period=20, std_mult=2.0):
    mid = sma(s, period)
    std = s.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std

def atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()


# === V1: Tight mean reversion (small target, tight stop) ===
def range_v1(df, bb_period=20, bb_std=2.0, rsi_p=14):
    """Buy lower band, sell at middle band. Stop at 1 ATR below entry."""
    mid, upper, lower = bollinger(df["close"], bb_period, bb_std)
    r = rsi(df["close"], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] <= lower) & (r < 35)] = 1
    sig[(df["close"] >= upper) & (r > 65)] = -1
    return sig


# === V2: RSI extreme only (simpler) ===
def range_v2(df, rsi_p=14, os=25, ob=75):
    """RSI extreme fade."""
    r = rsi(df["close"], rsi_p)
    sig = pd.Series(0, index=df.index)
    sig[(r < os) & (r.shift(1) >= os)] = 1
    sig[(r > ob) & (r.shift(1) <= ob)] = -1
    return sig


# === V3: Stochastic at extremes ===
def range_v3(df, k=14, d=3, os=20, ob=80):
    """Stochastic fade at extremes."""
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    raw_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    sk = raw_k.rolling(3).mean()
    sd = sk.rolling(d).mean()
    sig = pd.Series(0, index=df.index)
    sig[(sk < os) & (sk > sk.shift(1))] = 1
    sig[(sk > ob) & (sk < sk.shift(1))] = -1
    return sig


# === V4: EMA crossover (fast, short-term) ===
def range_v4(df, fast=5, slow=13):
    """Fast EMA crossover in range."""
    f = ema(df["close"], fast)
    s = ema(df["close"], slow)
    sig = pd.Series(0, index=df.index)
    sig[(f > s) & (f.shift(1) <= s.shift(1))] = 1
    sig[(f < s) & (f.shift(1) >= s.shift(1))] = -1
    return sig


# === V5: Price cross EMA (simple) ===
def range_v5(df, ema_p=20):
    """Price crosses EMA."""
    e = ema(df["close"], ema_p)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > e) & (df["close"].shift(1) <= e.shift(1))] = 1
    sig[(df["close"] < e) & (df["close"].shift(1) >= e.shift(1))] = -1
    return sig


# === V6: MACD histogram reversal ===
def range_v6(df, fast=12, slow=26, signal=9):
    """MACD histogram reversal."""
    macd_line = ema(df["close"], fast) - ema(df["close"], slow)
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    sig = pd.Series(0, index=df.index)
    sig[(hist > 0) & (hist.shift(1) <= 0)] = 1
    sig[(hist < 0) & (hist.shift(1) >= 0)] = -1
    return sig


# === V7: Inside bar + breakout ===
def range_v7(df):
    """Inside bar breakout."""
    is_inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    mother_high = df["high"].shift(1)
    mother_low = df["low"].shift(1)
    sig = pd.Series(0, index=df.index)
    prev_inside = is_inside.shift(1).fillna(False)
    sig[prev_inside & (df["close"] > mother_high)] = 1
    sig[prev_inside & (df["close"] < mother_low)] = -1
    return sig


# === V8: Doji/reversal candle ===
def range_v8(df):
    """Small body candle (indecision) followed by reversal."""
    body = abs(df["close"] - df["open"])
    wick = df["high"] - df["low"]
    is_doji = body < wick * 0.2  # Body < 20% of candle
    sig = pd.Series(0, index=df.index)
    # Bullish: doji at low, then green candle
    bull = is_doji.shift(1) & (df["close"] > df["open"]) & (df["low"] > df["low"].shift(1))
    # Bearish: doji at high, then red candle
    bear = is_doji.shift(1) & (df["close"] < df["open"]) & (df["high"] < df["high"].shift(1))
    sig[bull] = 1
    sig[bear] = -1
    return sig


# === V9: Keltner channel fade ===
def range_v9(df, ema_p=20, atr_p=14, mult=2.0):
    """Keltner channel fade."""
    mid = ema(df["close"], ema_p)
    a = atr(df, atr_p)
    upper = mid + mult * a
    lower = mid - mult * a
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < lower) & (df["close"].shift(1) >= lower.shift(1))] = 1
    sig[(df["close"] > upper) & (df["close"].shift(1) <= upper.shift(1))] = -1
    return sig


# === V10: Williams %R extreme ===
def range_v10(df, p=14, os=-80, ob=-20):
    """Williams %R fade."""
    high_max = df["high"].rolling(p).max()
    low_min = df["low"].rolling(p).min()
    w = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
    sig = pd.Series(0, index=df.index)
    sig[(w > os) & (w.shift(1) <= os)] = 1
    sig[(w < ob) & (w.shift(1) >= ob)] = -1
    return sig


ALL_RANGE_VARIANTS = {
    "range_v1_bb_rsi": range_v1,
    "range_v2_rsi": range_v2,
    "range_v3_stoch": range_v3,
    "range_v4_ema_cross": range_v4,
    "range_v5_ema_cross": range_v5,
    "range_v6_macd_hist": range_v6,
    "range_v7_inside_bar": range_v7,
    "range_v8_doji": range_v8,
    "range_v9_keltner": range_v9,
    "range_v10_williams": range_v10,
}
