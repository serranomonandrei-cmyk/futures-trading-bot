"""
Volume strategy development. Lean, fast, focused.
Tests VWMA + Volume Profile on all 10 coins.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ccxt
import numpy as np
import pandas as pd
from data import fetch_ohlcv
from config import PAIRS, TAKER_FEE_PCT, SLIPPAGE_PCT
from config import MIN_STOP_DISTANCE_PCT, MAX_MARGIN_UTILIZATION, STARTING_BALANCE

LEVERAGE = 15
RISK_PCT = 0.03
RR_RATIO = 4.0
ATR_STOP_MULT = 2.0
MAX_BARS_HELD = 72
MAX_CONCURRENT_POSITIONS = 10


def vwma_series(df, period):
    vp = df["close"] * df["volume"]
    return vp.rolling(period).sum() / df["volume"].rolling(period).sum()


def vwma_signal(df, period):
    s = vwma_series(df, period)
    cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
    cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cu] = 1; sig[cd] = -1
    return sig


def volprof_signal(df, period=20, steps=20, va_pct=0.68):
    poc_s = pd.Series(np.nan, index=df.index)
    vah_s = pd.Series(np.nan, index=df.index)
    val_s = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        w = df.iloc[i - period : i]
        pmin, pmax = w["low"].min(), w["high"].max()
        if pmax <= pmin: continue
        bs = (pmax - pmin) / steps
        buckets = np.zeros(steps)
        for _, r in w.iterrows():
            idx = min(int((r["close"] - pmin) / bs), steps - 1)
            buckets[idx] += r["volume"]
        total = buckets.sum()
        if total == 0: continue
        poc_i = np.argmax(buckets)
        cum = buckets[poc_i]; l, r_idx = poc_i - 1, poc_i + 1
        while cum / total < va_pct and (l >= 0 or r_idx < steps):
            if l >= 0 and (r_idx >= steps or buckets[l] >= buckets[r_idx]):
                cum += buckets[l]; l -= 1
            elif r_idx < steps:
                cum += buckets[r_idx]; r_idx += 1
            else: break
        vah_s.iloc[i] = pmin + (min(r_idx, steps - 1) + 0.5) * bs
        val_s.iloc[i] = pmin + (max(l + 1, 0) + 0.5) * bs
    sig = pd.Series(0, index=df.index)
    sig[df["close"] > vah_s] = 1
    sig[df["close"] < val_s] = -1
    return sig


def fast_backtest(df, signal_fn, rr=RR_RATIO, atr_mult=ATR_STOP_MULT, max_bars=MAX_BARS_HELD):
    """Minimal backtest with numpy for speed."""
    bal = float(STARTING_BALANCE)
    peak = bal
    trades = []

    # Precompute ATR (pandas, then convert to numpy)
    h, l, c_arr = df["high"].values, df["low"].values, df["close"].values
    close = df["close"]
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(), (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values

    signals = signal_fn(df).values
    n = len(df)

    # open_positions as list of dicts
    opens = []

    for i in range(1, n):
        o, h_i, l_i, c_i = df["open"].iloc[i], h[i], l[i], c_arr[i]

        # Check open positions
        still_open = []
        for p in opens:
            p["bars"] += 1
            hit = False; ex = 0.0; reason = ""
            if p["bars"] >= max_bars:
                ex, reason = c_i, "TIME"; hit = True
            elif p["side"] == "long":
                if l_i <= p["stop"]: ex, reason = p["stop"], "SL"; hit = True
                elif h_i >= p["tp"]: ex, reason = p["tp"], "TP"; hit = True
            else:
                if h_i >= p["stop"]: ex, reason = p["stop"], "SL"; hit = True
                elif l_i <= p["tp"]: ex, reason = p["tp"], "TP"; hit = True

            if hit:
                ret = (ex - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - ex) / p["entry"]
                ret -= SLIPPAGE_PCT
                gross = ret * (p["margin"] * LEVERAGE)
                fees = p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                bal += net
                peak = max(peak, bal)
                trades.append({"pnl": net, "side": p["side"], "bars": p["bars"], "reason": reason})
            else:
                still_open.append(p)
        opens = still_open

        # New signal (from previous bar close)
        sig = signals[i - 1]
        if sig != 0 and len(opens) < MAX_CONCURRENT_POSITIONS and bal > 0:
            atr_v = atr[i]
            if np.isnan(atr_v) or atr_v <= 0: continue

            side = "long" if sig == 1 else "short"
            entry = o * (1 + SLIPPAGE_PCT if sig == 1 else 1 - SLIPPAGE_PCT)

            dist = max(atr_v * atr_mult, entry * MIN_STOP_DISTANCE_PCT)
            stop = entry - dist if side == "long" else entry + dist
            tp = entry + dist * rr if side == "long" else entry - dist * rr

            risk_usd = bal * RISK_PCT
            stop_dist_pct = abs(entry - stop) / entry
            if stop_dist_pct < 1e-6: continue

            notional = min(risk_usd / stop_dist_pct, 500)
            margin = notional / LEVERAGE
            if margin < 1: continue

            total_m = sum(p["margin"] for p in opens)
            if total_m + margin > bal * MAX_MARGIN_UTILIZATION: continue

            opens.append({"side": side, "entry": entry, "stop": stop, "tp": tp, "margin": margin, "bars": 0})

    # Force close remaining
    last = c_arr[-1]
    for p in opens:
        ret = (last - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - last) / p["entry"]
        ret -= SLIPPAGE_PCT
        bal += ret * (p["margin"] * LEVERAGE) - p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    dd = (peak - bal) / peak if peak > 0 else 0
    return {"net_pnl": bal - STARTING_BALANCE, "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0, "max_drawdown": dd}


if __name__ == "__main__":
    print("\n=== VOLUME STRATEGY OPTIMIZATION ===\n")
    exchange = ccxt.binance({"enableRateLimit": True})
    all_results = {}
    periods = list(range(5, 61, 3))  # 5,8,11,...,59 = 19 periods

    for pair in PAIRS:
        coin = pair.split("/")[0]
        print(f"\n  {coin}: fetching data...", flush=True)
        df = fetch_ohlcv(exchange, pair, months=6)
        print(f"  {coin}: {len(df)} bars", flush=True)

        # Precompute all VWMA signals at once
        vp = df["close"] * df["volume"]
        signals = {}
        for p in periods:
            s = vp.rolling(p).sum() / df["volume"].rolling(p).sum()
            cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
            cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
            sig = pd.Series(0, index=df.index)
            sig[cu] = 1; sig[cd] = -1
            signals[p] = sig

        # Full backtest for each period using precomputed signals
        best_v = {"period": None, "net_pnl": -999}
        for p in periods:
            sig = signals[p]
            m = fast_backtest(df, lambda d, s=sig: s)
            if m["net_pnl"] > best_v["net_pnl"]:
                best_v = {"period": p, **m}
        arrow = "+" if best_v["net_pnl"] > 0 else ""
        print(f"  VWMA({best_v['period']})  PnL: ${arrow}{best_v['net_pnl']:>.2f}  WR: {best_v['win_rate']:.0%}  Trades: {best_v['total_trades']}", flush=True)

        # Volume Profile
        print("  VOLPROF: computing signal...", end=" ", flush=True)
        vp_sig = volprof_signal(df)
        print("done, backtesting...", end=" ", flush=True)
        m_vp = fast_backtest(df, lambda d: vp_sig)
        print("done", flush=True)
        arrow2 = "+" if m_vp["net_pnl"] > 0 else ""
        print(f"  VOLPROF(20)   PnL: ${arrow2}{m_vp['net_pnl']:>.2f}  WR: {m_vp['win_rate']:.0%}  Trades: {m_vp['total_trades']}", flush=True)

        all_results[coin] = {"VWMA": best_v, "VOLPROF": m_vp}

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for sname in ["VWMA", "VOLPROF"]:
        print(f"\n  --- {sname} ---")
        prof = sum(1 for v in all_results.values() if v[sname]["net_pnl"] > 0)
        tot = sum(v[sname]["net_pnl"] for v in all_results.values())
        for c, v in sorted(all_results.items()):
            r = v[sname]
            per = f"({r['period']})" if "period" in r and r["period"] else ""
            arrow = "+" if r["net_pnl"] > 0 else ""
            print(f"    {c:6s} {sname}{per:4s}  PnL: ${arrow}{r['net_pnl']:>8.2f}  WR: {r['win_rate']:.0%}  Trades: {r['total_trades']}")
        print(f"    → Profitable: {prof}/{len(all_results)} | Total: ${tot:+.2f}")

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out}")
