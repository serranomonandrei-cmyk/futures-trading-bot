"""
20 strategies. Each: df -> signal column (1=long, -1=short, 0=flat).
No look-ahead. Signal on bar N close, acts on bar N+1.
"""

import numpy as np
import pandas as pd


# === helpers ===

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

def stoch(df, k=14, d=3, smooth=3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    raw = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    return raw.rolling(smooth).mean(), raw.rolling(smooth).mean().rolling(d).mean()

def williams_r(df, p=14):
    high_max = df["high"].rolling(p).max()
    low_min = df["low"].rolling(p).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

def cci(df, p=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp - tp.rolling(p).mean()) / (0.015 * tp.rolling(p).std())

def roc(s, p=10):
    return (s / s.shift(p) - 1) * 100

def hma(s, p=20):
    w = ema(s, p//2) * 2 - ema(s, p)
    return ema(w, int(np.sqrt(p)))


# === TREND FOLLOWING ===

def ema_crossover(df, fast=9, slow=21):
    """EMA 9 crosses above 21 = long, below = short."""
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    sig = pd.Series(0, index=df.index)
    sig[(f > s) & (f.shift(1) <= s.shift(1))] = 1
    sig[(f < s) & (f.shift(1) >= s.shift(1))] = -1
    return sig

def triple_ema(df, fast=8, mid=21, slow=55):
    """All 3 aligned: fast>mid>slow = long, fast<mid<mid = short."""
    f, m, s = ema(df["close"], fast), ema(df["close"], mid), ema(df["close"], slow)
    sig = pd.Series(0, index=df.index)
    sig[(f > m) & (m > s) & ~((f.shift(1) > m.shift(1)) & (m.shift(1) > s.shift(1)))] = 1
    sig[(f < m) & (m < s) & ~((f.shift(1) < m.shift(1)) & (m.shift(1) < s.shift(1)))] = -1
    return sig

def donchian_breakout(df, period=20):
    """Break above 20-period high = long, below low = short. Only on breakout bar."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    sig = pd.Series(0, index=df.index)
    # Only signal on the bar that FIRST breaks the level
    above = df["close"] > upper.shift(1)
    below = df["close"] < lower.shift(1)
    sig[above & ~above.shift(1).fillna(False)] = 1
    sig[below & ~below.shift(1).fillna(False)] = -1
    return sig

def adx_di(df, period=14, adx_thresh=25):
    """ADX > 25 with +DI > -DI = long, -DI > +DI = short."""
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(period).mean()
    plus_di = 100 * ema(plus_dm, period) / (atr14 + 1e-10)
    minus_di = 100 * ema(minus_dm, period) / (atr14 + 1e-10)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx = ema(dx, period)
    sig = pd.Series(0, index=df.index)
    sig[(adx > adx_thresh) & (plus_di > minus_di) & (adx.shift(1) <= adx_thresh)] = 1
    sig[(adx > adx_thresh) & (minus_di > plus_di) & (adx.shift(1) <= adx_thresh)] = -1
    return sig

def parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    """Parabolic SAR flip."""
    high, low = df["high"].values, df["low"].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.ones(n)  # 1=up, -1=down
    ep = high[0]
    af = af_start
    sar[0] = low[0]

    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend[i-1] == 1:
            sar[i] = min(sar[i], low[i-1], low[max(0,i-2)])
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = max(sar[i], high[i-1], high[max(0,i-2)])
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    sig = pd.Series(0, index=df.index)
    trend_s = pd.Series(trend, index=df.index)
    sig[(trend_s == 1) & (trend_s.shift(1) == -1)] = 1
    sig[(trend_s == -1) & (trend_s.shift(1) == 1)] = -1
    return sig


# === MEAN REVERSION ===

def rsi_extreme(df, period=14, oversold=30, overbought=70):
    """RSI crosses above oversold = long, below overbought = short."""
    r = rsi(df["close"], period)
    sig = pd.Series(0, index=df.index)
    sig[(r > oversold) & (r.shift(1) <= oversold)] = 1
    sig[(r < overbought) & (r.shift(1) >= overbought)] = -1
    return sig

def bollinger_bounce(df, period=20, std_mult=2.0):
    """Touch lower band = long, touch upper band = short."""
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    sig = pd.Series(0, index=df.index)
    sig[df["close"] <= lower] = 1
    sig[df["close"] >= upper] = -1
    return sig

def keltner_fade(df, ema_p=20, atr_p=14, mult=2.0):
    """Price outside Keltner Channel = fade back to mean."""
    mid = ema(df["close"], ema_p)
    a = atr(df, atr_p)
    upper = mid + mult * a
    lower = mid - mult * a
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < lower) & (df["close"].shift(1) >= lower)] = 1
    sig[(df["close"] > upper) & (df["close"].shift(1) <= upper)] = -1
    return sig

def stochastic_extreme(df, k=14, d=3, oversold=20, overbought=80):
    """Stoch %K crosses above %D at oversold = long, vice versa at overbought."""
    sk, sd = stoch(df, k, d)
    sig = pd.Series(0, index=df.index)
    sig[(sk > sd) & (sk.shift(1) <= sd.shift(1)) & (sk < oversold + 20)] = 1
    sig[(sk < sd) & (sk.shift(1) >= sd.shift(1)) & (sk > overbought - 20)] = -1
    return sig

def williams_r_extreme(df, period=14, oversold=-80, overbought=-20):
    """Williams %R crosses above oversold = long, below overbought = short."""
    w = williams_r(df, period)
    sig = pd.Series(0, index=df.index)
    sig[(w > oversold) & (w.shift(1) <= oversold)] = 1
    sig[(w < overbought) & (w.shift(1) >= overbought)] = -1
    return sig


# === BREAKOUT ===

def volume_breakout(df, vol_mult=2.0, price_pct=0.005):
    """Volume > 2x avg AND price moves > 0.5% = breakout. Only on first bar."""
    vol_avg = df["volume"].rolling(20).mean()
    pct_change = df["close"].pct_change()
    vol_spike = df["volume"] > vol_avg * vol_mult
    up = vol_spike & (pct_change > price_pct)
    dn = vol_spike & (pct_change < -price_pct)
    sig = pd.Series(0, index=df.index)
    sig[up & ~up.shift(1).fillna(False)] = 1
    sig[dn & ~dn.shift(1).fillna(False)] = -1
    return sig

def atr_breakout(df, period=14, mult=1.5):
    """Price breaks above prev close + ATR*mult = long, below = short."""
    a = atr(df, period)
    sig = pd.Series(0, index=df.index)
    sig[df["close"] > df["close"].shift(1) + a.shift(1) * mult] = 1
    sig[df["close"] < df["close"].shift(1) - a.shift(1) * mult] = -1
    return sig

def inside_bar_breakout(df):
    """Inside bar followed by breakout of mother bar high/low."""
    is_inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    mother_high = df["high"].shift(1)
    mother_low = df["low"].shift(1)
    sig = pd.Series(0, index=df.index)
    prev_inside = is_inside.shift(1).fillna(False)
    sig[prev_inside & (df["close"] > mother_high)] = 1
    sig[prev_inside & (df["close"] < mother_low)] = -1
    return sig

def range_breakout(df, period=20):
    """Break above N-period high = long, below low = short. Only on breakout bar."""
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    sig = pd.Series(0, index=df.index)
    above = df["close"] > highest.shift(1)
    below = df["close"] < lowest.shift(1)
    sig[above & ~above.shift(1).fillna(False)] = 1
    sig[below & ~below.shift(1).fillna(False)] = -1
    return sig

def pivot_breakout(df):
    """Price breaks previous day pivot point."""
    # Use daily pivot (approximated from 1H by taking every 24th bar)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pivot = tp.shift(24)  # ~1 day on 1H
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > pivot) & (df["close"].shift(1) <= pivot.shift(1))] = 1
    sig[(df["close"] < pivot) & (df["close"].shift(1) >= pivot.shift(1))] = -1
    return sig


# === MOMENTUM ===

def macd_crossover(df, fast=12, slow=26, signal=9):
    """MACD line crosses signal line."""
    macd_line = ema(df["close"], fast) - ema(df["close"], slow)
    sig_line = ema(macd_line, signal)
    sig = pd.Series(0, index=df.index)
    sig[(macd_line > sig_line) & (macd_line.shift(1) <= sig_line.shift(1))] = 1
    sig[(macd_line < sig_line) & (macd_line.shift(1) >= sig_line.shift(1))] = -1
    return sig

def roc_momentum(df, period=10, threshold=0.5):
    """ROC crosses above threshold = long, below = short."""
    r = roc(df["close"], period)
    sig = pd.Series(0, index=df.index)
    sig[(r > threshold) & (r.shift(1) <= threshold)] = 1
    sig[(r < -threshold) & (r.shift(1) >= -threshold)] = -1
    return sig

def hull_ma_crossover(df, period=20):
    """Hull MA crosses price."""
    h = hma(df["close"], period)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > h) & (df["close"].shift(1) <= h.shift(1))] = 1
    sig[(df["close"] < h) & (df["close"].shift(1) >= h.shift(1))] = -1
    return sig

def cci_extreme(df, period=20, oversold=-100, overbought=100):
    """CCI crosses above oversold = long, below overbought = short."""
    c = cci(df, period)
    sig = pd.Series(0, index=df.index)
    sig[(c > oversold) & (c.shift(1) <= oversold)] = 1
    sig[(c < overbought) & (c.shift(1) >= overbought)] = -1
    return sig


# === MULTI-SIGNAL ===

def ema_rsi_volume(df, ema_p=21, rsi_p=14, vol_mult=1.5):
    """EMA trend + RSI confirmation + volume above average."""
    e = ema(df["close"], ema_p)
    r = rsi(df["close"], rsi_p)
    vol_avg = df["volume"].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    long = (df["close"] > e) & (r > 40) & (r < 70) & (df["volume"] > vol_avg * vol_mult)
    short = (df["close"] < e) & (r < 60) & (r > 30) & (df["volume"] > vol_avg * vol_mult)
    sig[long & ~long.shift(1).fillna(False)] = 1
    sig[short & ~short.shift(1).fillna(False)] = -1
    return sig


# === REGISTRY ===

ALL_STRATEGIES = {
    # Trend Following
    "ema_cross_9_21": lambda df: ema_crossover(df, 9, 21),
    "ema_cross_5_13": lambda df: ema_crossover(df, 5, 13),
    "triple_ema": triple_ema,
    "donchian_20": lambda df: donchian_breakout(df, 20),
    "adx_di": adx_di,
    "parabolic_sar": parabolic_sar,
    # Mean Reversion
    "rsi_extreme": rsi_extreme,
    "bollinger_bounce": bollinger_bounce,
    "keltner_fade": keltner_fade,
    "stochastic_extreme": stochastic_extreme,
    "williams_r": williams_r_extreme,
    # Breakout
    "volume_breakout": volume_breakout,
    "atr_breakout": atr_breakout,
    "inside_bar": inside_bar_breakout,
    "range_breakout_20": lambda df: range_breakout(df, 20),
    # Momentum
    "macd_cross": macd_crossover,
    "roc_momentum": roc_momentum,
    "hull_ma": hull_ma_crossover,
    "cci_extreme": cci_extreme,
    # Multi-Signal
    "ema_rsi_volume": ema_rsi_volume,
}

def list_strategies():
    return list(ALL_STRATEGIES.keys())

def get_strategy(name):
    return ALL_STRATEGIES[name]
