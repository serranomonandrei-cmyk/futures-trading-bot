"""
Multi-coin VWMA backtest — shared $20, all 10 coins, per-coin optimal periods.
Matches bot_multi.py structure but uses VWMA strategies.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ccxt
import pandas as pd
import numpy as np
from data import fetch_ohlcv
from config import TAKER_FEE_PCT, SLIPPAGE_PCT, MIN_STOP_DISTANCE_PCT
from config import MAX_MARGIN_UTILIZATION, STARTING_BALANCE

LEVERAGE = 15
RISK_PCT = 0.03
RR_RATIO = 4.0
ATR_STOP_MULT = 2.0
MAX_BARS_HELD = 72
MAX_POSITIONS = 10

# Per-coin optimized VWMA periods (from volume.py backtest)
VWMA_STRATEGIES = {
    "ETH/USDT:USDT": 59,
    "BTC/USDT:USDT": 23,
    "SOL/USDT:USDT": 32,
    "BNB/USDT:USDT": 47,
    "SUI/USDT:USDT": 23,
    "LINK/USDT:USDT": 26,
    "ADA/USDT:USDT": 56,
    "DOGE/USDT:USDT": 41,
    "XRP/USDT:USDT": 8,
    "BCH/USDT:USDT": 26,
}


def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def vwma_signal(df, period):
    vp = df["close"] * df["volume"]
    s = vp.rolling(period).sum() / df["volume"].rolling(period).sum()
    cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
    cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cu] = 1; sig[cd] = -1
    return sig


def backtest_multi():
    exchange = ccxt.binance({"enableRateLimit": True})
    pairs = list(VWMA_STRATEGIES.keys())
    fund = STARTING_BALANCE
    peak = fund
    trades = []
    coin_data = {}

    # Fetch and precompute signals for all coins
    for pair in pairs:
        coin = pair.split("/")[0]
        p = VWMA_STRATEGIES[pair]
        print(f"  {coin}: VWMA({p})...", end=" ", flush=True)
        df = fetch_ohlcv(exchange, pair, months=6)
        sig = vwma_signal(df, p).values
        atr = calc_atr(df).values
        coin_data[coin] = {
            "open": df["open"].values, "high": df["high"].values,
            "low": df["low"].values, "close": df["close"].values,
            "sig": sig, "atr": atr, "n": len(df),
        }
        print(f"{len(df)} bars", flush=True)

    n_bars = max(c["n"] for c in coin_data.values())
    opens = []  # list of active positions

    for i in range(1, n_bars):
        # Check positions
        still_open = []
        for p in opens:
            coin = p["coin"]
            d = coin_data[coin]
            if i >= d["n"]:
                continue
            hi, lo, cl = d["high"][i], d["low"][i], d["close"][i]
            p["bars"] += 1
            hit = False; ex = 0.0; reason = ""
            if p["bars"] >= MAX_BARS_HELD:
                ex, reason = cl, "TIME"; hit = True
            elif p["side"] == "long":
                if lo <= p["stop"]: ex, reason = p["stop"], "SL"; hit = True
                elif hi >= p["tp"]: ex, reason = p["tp"], "TP"; hit = True
            else:
                if hi >= p["stop"]: ex, reason = p["stop"], "SL"; hit = True
                elif lo <= p["tp"]: ex, reason = p["tp"], "TP"; hit = True
            if hit:
                ret = (ex - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - ex) / p["entry"]
                ret -= SLIPPAGE_PCT
                gross = ret * (p["margin"] * LEVERAGE)
                fees = p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                fund += net
                peak = max(peak, fund)
                trades.append({"coin": coin, "pnl": net, "side": p["side"], "bars": p["bars"], "reason": reason})
            else:
                still_open.append(p)
        opens = still_open

        # New signals (from prev bar)
        for coin, d in coin_data.items():
            if i >= d["n"]: continue
            if len(opens) >= MAX_POSITIONS: break
            sig = d["sig"][i-1]
            if sig == 0: continue

            # Check if already in position for this coin
            if any(p["coin"] == coin for p in opens): continue
            if fund <= 0: continue

            atr_v = d["atr"][i]
            if np.isnan(atr_v) or atr_v <= 0: continue

            side = "long" if sig == 1 else "short"
            entry = d["open"][i] * (1 + SLIPPAGE_PCT if sig == 1 else 1 - SLIPPAGE_PCT)
            dist = max(atr_v * ATR_STOP_MULT, entry * MIN_STOP_DISTANCE_PCT)
            stop = entry - dist if side == "long" else entry + dist
            tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO

            risk_usd = fund * RISK_PCT
            stop_dist_pct = abs(entry - stop) / entry
            if stop_dist_pct < 1e-6: continue

            notional = min(risk_usd / stop_dist_pct, 500)
            margin = notional / LEVERAGE
            if margin < 1: continue

            total_m = sum(p["margin"] for p in opens)
            if total_m + margin > fund * MAX_MARGIN_UTILIZATION: continue

            opens.append({"coin": coin, "side": side, "entry": entry, "stop": stop, "tp": tp, "margin": margin, "bars": 0})

    # Force close remaining
    for p in opens:
        coin, d = p["coin"], coin_data[p["coin"]]
        last = d["close"][d["n"] - 1]
        ret = (last - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - last) / p["entry"]
        ret -= SLIPPAGE_PCT
        fund += ret * (p["margin"] * LEVERAGE) - p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2

    # Per-coin PnL
    coin_pnl = {}
    for t in trades:
        coin_pnl[t["coin"]] = coin_pnl.get(t["coin"], 0) + t["pnl"]

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    wr = len(wins) / len(trades) if trades else 0
    dd = (peak - fund) / peak if peak > 0 else 0

    print("\n" + "=" * 60)
    print(f"  Final: \${fund:.2f} | PnL: \${fund-STARTING_BALANCE:+.2f} | Trades: {len(trades)} | WR: {wr:.0%} | DD: {dd:.0%}")
    print("=" * 60)
    for c in sorted(coin_pnl, key=lambda c: coin_pnl[c], reverse=True):
        arrow = "+" if coin_pnl[c] > 0 else ""
        print(f"    {c:6s}: \${arrow}{coin_pnl[c]:>8.2f}")
    prof = sum(1 for v in coin_pnl.values() if v > 0)
    print(f"  → Profitable: {prof}/{len(coin_data)}")

    return {"final": fund, "pnl": fund - STARTING_BALANCE, "trades": len(trades), "wr": wr, "dd": dd, "coin_pnl": coin_pnl}


if __name__ == "__main__":
    print("\n=== VWMA MULTI-COIN BACKTEST (shared $20) ===\n")
    r = backtest_multi()
