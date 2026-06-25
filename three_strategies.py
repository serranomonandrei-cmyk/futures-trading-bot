"""
Strategy library v2. Systematic search. No Donchian. New ideas.
Each strategy takes df (must have 'regime' column) and returns signals.
"""

import numpy as np
import pandas as pd


def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()
def _rsi(s, p=14):
    d = s.diff(); u = d.clip(0).rolling(p).mean(); dn = -d.clip(upper=0).rolling(p).mean()
    return 100 - 100 / (1 + u / dn)
def _atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1).rolling(p).mean()


# ============================================================
# ARCHETYPE 1: Volatility Contraction Pattern (VCP)
# ============================================================
def vcp_breakout(df, lookback=20, contraction_pct=0.5, vol_pct=0.7, entry_period=10):
    """Tightest range in N bars + falling volume + breakout.
    contraction_pct: current ATR < contraction_pct * max ATR in lookback
    vol_pct: current volume < vol_pct * avg volume"""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    for regime, direction in [("trending_up", 1), ("trending_down", -1)]:
        mask = df["regime"] == regime
        if mask.sum() < 30: continue

        a = _atr(df, 14)
        max_a = a.rolling(lookback).max()
        v_avg = df["volume"].rolling(20).mean()

        contracting = (a < max_a * contraction_pct) & (df["volume"] < v_avg * vol_pct)

        if direction == 1:
            hi = df["high"].rolling(entry_period).max()
            sig[mask & contracting & (df["close"] > hi.shift(1))] = 1
        else:
            lo = df["low"].rolling(entry_period).min()
            sig[mask & contracting & (df["close"] < lo.shift(1))] = -1

    # Ranging: VCP breakout either direction
    mask_r = df["regime"] == "ranging"
    if mask_r.sum() >= 30:
        a = _atr(df, 14); max_a = a.rolling(lookback).max()
        v_avg = df["volume"].rolling(20).mean()
        contracting = (a < max_a * contraction_pct) & (df["volume"] < v_avg * vol_pct)
        hi = df["high"].rolling(entry_period).max()
        lo = df["low"].rolling(entry_period).min()
        sig[mask_r & contracting & (df["close"] > hi.shift(1))] = 1
        sig[mask_r & contracting & (df["close"] < lo.shift(1))] = -1

    return sig


# ============================================================
# ARCHETYPE 2: Opening Range Breakout (ORB)
# ============================================================
def orb_breakout(df, orb_hours=4):
    """First N hours of each UTC day define the range. Breakout = entry.
    Works on 1H data: orb_hours=4 means first 4 hours (00:00-04:00 UTC)."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    if "timestamp" not in df.columns: return sig
    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.date

    # Calculate daily orbital range
    for d in day.unique():
        day_mask = day == d
        day_idxs = df.index[day_mask]
        if len(day_idxs) < orb_hours + 1: continue

        orb_end = day_idxs[orb_hours - 1]
        orb_high = df["high"].iloc[day_idxs[0]:day_idxs[orb_hours-1]+1].max()
        orb_low = df["low"].iloc[day_idxs[0]:day_idxs[orb_hours-1]+1].min()

        post_orb = day_idxs[orb_hours:]
        for idx in post_orb:
            regime = df["regime"].iloc[idx]
            if regime in ["trending_up", "ranging"] and df["close"].iloc[idx] > orb_high:
                sig.iloc[idx] = 1
            elif regime in ["trending_down", "ranging"] and df["close"].iloc[idx] < orb_low:
                sig.iloc[idx] = -1

    return sig


# ============================================================
# ARCHETYPE 3: EMA Ribbon (aligned MAs)
# ============================================================
def ema_ribbon(df, periods=[5, 10, 20, 50], min_aligned=4):
    """All EMAs aligned = strong trend. Enter on continuation."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    emas = {p: _ema(df["close"], p) for p in periods}

    for regime, direction in [("trending_up", 1), ("trending_down", -1)]:
        mask = df["regime"] == regime
        if mask.sum() < 50: continue

        if direction == 1:
            aligned = pd.Series(True, index=df.index)
            for i in range(len(periods)-1):
                aligned &= emas[periods[i]] > emas[periods[i+1]]
            # Enter on green candle when aligned
            green = df["close"] > df["open"]
            sig[mask & aligned & green] = 1
        else:
            aligned = pd.Series(True, index=df.index)
            for i in range(len(periods)-1):
                aligned &= emas[periods[i]] < emas[periods[i+1]]
            red = df["close"] < df["open"]
            sig[mask & aligned & red] = -1

    return sig


# ============================================================
# ARCHETYPE 4: Volume Climax Reversal
# ============================================================
def volume_climax(df, vol_mult=3.0, body_atr_mult=1.5):
    """Extreme volume + large candle → exhaustion. Enter reverse next bar."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    a = _atr(df, 14)
    v_avg = df["volume"].rolling(20).mean()
    body = abs(df["close"] - df["open"])
    vol_spike = df["volume"] > v_avg * vol_mult
    large_body = body > a * body_atr_mult

    for regime, direction in [("trending_up", -1), ("trending_down", 1)]:
        mask = df["regime"] == regime
        if mask.sum() < 30: continue
        if direction == 1:
            # Big red candle in trending_down → exhausted, go long
            big_red = (df["close"] < df["open"]) & vol_spike & large_body
            sig[mask & big_red] = 1
        else:
            # Big green candle in trending_up → exhausted, go short
            big_green = (df["close"] > df["open"]) & vol_spike & large_body
            sig[mask & big_green] = -1

    return sig


# ============================================================
# ARCHETYPE 5: ADX Trend Start (ADX rising from low)
# ============================================================
def adx_trend_start(df, adx_p=14, adx_low=20, adx_high=30, entry_p=10):
    """ADX rising from <20 to >25+ means trend starting. Enter."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.rolling(adx_p).mean()
    up = h - h.shift(1); dn = l.shift(1) - l
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=df.index)
    pdi = 100 * plus_dm.rolling(adx_p).mean() / a
    mdi = 100 * minus_dm.rolling(adx_p).mean() / a
    dx = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
    adx = dx.rolling(adx_p).mean()

    was_low = (adx.shift(3) < adx_low) & (adx > adx_low)
    trend_up = pdi > mdi
    trend_down = mdi > pdi

    for regime, direction in [("trending_up", 1), ("trending_down", -1)]:
        mask = df["regime"] == regime
        if mask.sum() < 30: continue
        if direction == 1:
            hi = df["high"].rolling(entry_p).max()
            sig[mask & was_low & trend_up & (df["close"] > hi.shift(1))] = 1
        else:
            lo = df["low"].rolling(entry_p).min()
            sig[mask & was_low & trend_down & (df["close"] < lo.shift(1))] = -1

    return sig


# ============================================================
# ARCHETYPE 6: Bollinger Band Walk (riding the bands)
# ============================================================
def bb_walk(df, bb_p=20, bb_std=2.0):
    """Price riding upper/lower BB with band expansion → strong trend entry."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    mid = _sma(df["close"], bb_p)
    std = df["close"].rolling(bb_p).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    bb_width = (upper - lower) / mid
    expanding = bb_width > bb_width.shift(5)

    for regime, direction in [("trending_up", 1), ("trending_down", -1)]:
        mask = df["regime"] == regime
        if mask.sum() < 30: continue
        if direction == 1:
            walking = (df["close"] > upper.shift(1)) & expanding
            sig[mask & walking.shift(1) & (df["close"] > df["open"])] = 1
        else:
            walking = (df["close"] < lower.shift(1)) & expanding
            sig[mask & walking.shift(1) & (df["close"] < df["open"])] = -1

    return sig


# ============================================================
# ARCHETYPE 7: Inside Bar Stack → Breakout
# ============================================================
def inside_bar_breakout(df, min_inside=2):
    """Multiple inside bars → breakout of mother's range."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))

    # Count consecutive inside bars
    blocks = (inside != inside.shift()).fillna(True).cumsum()
    runs = inside.astype(int).groupby(blocks).cumsum()

    # Mother bar = last bar before inside bars started
    # Simple approach: if we have N consecutive inside bars, the mother is N+1 bars ago
    stacked = runs >= min_inside

    for regime, direction in [("trending_up", 1), ("trending_down", -1), ("ranging", 1), ("ranging", -1)]:
        mask = df["regime"] == regime
        if mask.sum() < 20: continue
        if direction == 1:
            mother_high = df["high"].shift(min_inside)
            sig[mask & stacked & (df["close"] > mother_high)] = 1
        else:
            mother_low = df["low"].shift(min_inside)
            sig[mask & stacked & (df["close"] < mother_low)] = -1

    return sig


# ============================================================
# ARCHETYPE 8: Keltner Channel Mean Reversion (improved)
# ============================================================
def keltner_mr(df, ema_p=20, atr_p=14, atr_mult=2.0, rsi_os=30, rsi_ob=70):
    """Keltner channel fade with RSI confirmation."""
    sig = pd.Series(0, index=df.index)
    if "regime" not in df.columns: return sig

    kc_mid = _ema(df["close"], ema_p)
    a = _atr(df, atr_p)
    kc_upper = kc_mid + atr_mult * a
    kc_lower = kc_mid - atr_mult * a
    r = _rsi(df["close"], 14)

    for regime, os_val, ob_val in [("trending_up", 35, 70), ("trending_down", 30, 65), ("ranging", 25, 75)]:
        mask = df["regime"] == regime
        if mask.sum() < 20: continue
        sig[mask & (df["close"] < kc_lower) & (r < os_val)] = 1
        sig[mask & (df["close"] > kc_upper) & (r > ob_val)] = -1

    return sig


# Registry — all strategies to test
ALL_STRATEGIES = {
    "vcp_basic": vcp_breakout,
    "orb_daily": orb_breakout,
    "ema_ribbon": ema_ribbon,
    "vol_climax": volume_climax,
    "adx_trend": adx_trend_start,
    "bb_walk": bb_walk,
    "inside_bar": inside_bar_breakout,
    "keltner_mr": keltner_mr,
}

# Parameter grid to search over
VCP_PARAMS = [
    {"lookback": 20, "contraction_pct": 0.5, "vol_pct": 0.7, "entry_period": 10},
    {"lookback": 30, "contraction_pct": 0.4, "vol_pct": 0.6, "entry_period": 20},
    {"lookback": 50, "contraction_pct": 0.3, "vol_pct": 0.5, "entry_period": 20},
]
ORB_PARAMS = [{"orb_hours": 4}, {"orb_hours": 8}, {"orb_hours": 12}]
RIBBON_PARAMS = [
    {"periods": [5,10,20,50]},
    {"periods": [10,20,50,100]},
    {"periods": [5,10,20]},
]
CLIMAX_PARAMS = [
    {"vol_mult": 3.0, "body_atr_mult": 1.5},
    {"vol_mult": 4.0, "body_atr_mult": 2.0},
]
ADX_PARAMS = [
    {"adx_p": 14, "adx_low": 20, "adx_high": 30, "entry_p": 10},
    {"adx_p": 14, "adx_low": 15, "adx_high": 25, "entry_p": 20},
]
BBW_PARAMS = [
    {"bb_p": 20, "bb_std": 2.0},
    {"bb_p": 20, "bb_std": 2.5},
    {"bb_p": 30, "bb_std": 2.0},
]
IB_PARAMS = [
    {"min_inside": 2},
    {"min_inside": 3},
    {"min_inside": 4},
]
KELTNER_PARAMS = [
    {"ema_p": 20, "atr_p": 14, "atr_mult": 2.0},
    {"ema_p": 20, "atr_p": 14, "atr_mult": 2.5},
    {"ema_p": 30, "atr_p": 14, "atr_mult": 2.0},
]

STRAT_PARAMS = {
    "vcp_basic": VCP_PARAMS,
    "orb_daily": ORB_PARAMS,
    "ema_ribbon": RIBBON_PARAMS,
    "vol_climax": CLIMAX_PARAMS,
    "adx_trend": ADX_PARAMS,
    "bb_walk": BBW_PARAMS,
    "inside_bar": IB_PARAMS,
    "keltner_mr": KELTNER_PARAMS,
}