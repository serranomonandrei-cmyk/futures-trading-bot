"""
Test strategy improvements based on trade analysis findings.
Tests: wider SL, breakeven stop, short bias, no-SL grace period.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ccxt
import pandas as pd
import numpy as np
from data import fetch_ohlcv
from config import TAKER_FEE_PCT, SLIPPAGE_PCT, MIN_STOP_DISTANCE_PCT
from config import MAX_MARGIN_UTILIZATION, STARTING_BALANCE
from bot_multi import STRATEGIES, calc_signal, calc_atr

LEVERAGE = 15; RISK_PCT = 0.03
MAX_BARS_HELD = 72; MAX_POSITIONS = 10


def prepare_data():
    """Fetch all data once, return dicts."""
    exchange = ccxt.binance({"enableRateLimit": True})
    data = {}
    for pair in STRATEGIES:
        data[pair] = fetch_ohlcv(exchange, pair, months=6)
        print(f"  {pair.split('/')[0]}: {len(data[pair])} bars")
    sigs, atrs = {}, {}
    for pair, df in data.items():
        s = pd.Series(0, index=df.index)
        for i in range(len(df)):
            s.iloc[i] = calc_signal(df.iloc[:i+1], pair)
        sigs[pair] = s
        atrs[pair] = calc_atr(df)
    return data, sigs, atrs


def backtest(data, sigs, atrs, rr=4.0, atr_mult=2.0, breakeven_bars=0, short_rr=None, grace_bars=0):
    """Run multi-coin backtest with tuneable params."""
    bal = float(STARTING_BALANCE)
    peak = bal
    positions = {}
    trades = []
    master = data["ETH/USDT:USDT"]
    n = len(master)
    loss_streak = 0

    for i in range(1, n):
        ts = master["timestamp"].iloc[i]

        # Close
        for pair in list(positions.keys()):
            pdf = data[pair]; row_mask = pdf["timestamp"] == ts
            if not row_mask.any(): continue
            row = pdf[row_mask].iloc[0]
            pos = positions[pair]; pos["bars"] += 1

            exit_p = None; reason = None

            # Breakeven: if trade survived > breakeven_bars, move stop to entry
            if breakeven_bars > 0 and pos["bars"] >= breakeven_bars:
                pos["stop"] = pos["entry"]

            # Grace: no SL in first grace_bars
            sl_active = pos["bars"] >= grace_bars

            if pos["bars"] >= MAX_BARS_HELD:
                exit_p = row["close"]; reason = "TIME"
            elif pos["side"] == "long":
                if sl_active and row["low"] <= pos["stop"]: exit_p = pos["stop"]; reason = "SL"
                elif row["high"] >= pos["tp"]: exit_p = pos["tp"]; reason = "TP"
            else:
                if sl_active and row["high"] >= pos["stop"]: exit_p = pos["stop"]; reason = "SL"
                elif row["low"] <= pos["tp"]: exit_p = pos["tp"]; reason = "TP"

            if exit_p is not None:
                pnl_pct = (exit_p - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - exit_p) / pos["entry"] - SLIPPAGE_PCT
                gross = pnl_pct * (pos["margin"] * LEVERAGE)
                fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                bal += net
                trades.append({"coin": pair.split("/")[0], "pnl": net, "reason": reason, "side": pos["side"]})
                del positions[pair]

        # Open
        if len(positions) < MAX_POSITIONS and bal > 0:
            for pair in STRATEGIES:
                if pair in positions or len(positions) >= MAX_POSITIONS: continue
                pdf = data[pair]; row_mask = pdf["timestamp"] == ts
                if not row_mask.any(): continue
                idx = pdf[row_mask].index[0]
                if idx not in sigs[pair].index or sigs[pair].loc[idx] == 0: continue
                row = pdf.loc[idx]
                side = "long" if sigs[pair].loc[idx] == 1 else "short"
                entry = row["open"] * (1 + SLIPPAGE_PCT if side == "long" else 1 - SLIPPAGE_PCT)
                atr_val = atrs[pair].loc[idx] if idx in atrs[pair].index else 0
                if pd.isna(atr_val) or atr_val <= 0: continue
                dist = max(atr_val * atr_mult, entry * MIN_STOP_DISTANCE_PCT)
                stop = entry - dist if side == "long" else entry + dist
                tp_rr = short_rr if (short_rr and side == "short") else rr
                tp = entry + dist * tp_rr if side == "long" else entry - dist * tp_rr
                risk_usd = bal * RISK_PCT
                stop_dist_pct = abs(entry - stop) / entry
                if stop_dist_pct < 1e-6: continue
                notional = min(risk_usd / stop_dist_pct, 500)
                margin = notional / LEVERAGE
                if margin < 1.0: continue
                total_margin = sum(p["margin"] for p in positions.values())
                if total_margin + margin > bal * MAX_MARGIN_UTILIZATION: continue
                positions[pair] = {"side": side, "entry": entry, "stop": stop, "tp": tp,
                                   "margin": margin, "bars": 0}
                peak = max(peak, bal)

    for pair in list(positions.keys()):
        last = data[pair].iloc[-1]; pos = positions[pair]
        pnl_pct = (last["close"] - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - last["close"]) / pos["entry"] - SLIPPAGE_PCT
        gross = pnl_pct * (pos["margin"] * LEVERAGE)
        fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
        net = gross - fees
        bal += net
        trades.append({"coin": pair.split("/")[0], "pnl": net, "reason": "FORCE", "side": pos["side"]})

    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "final": bal, "pnl": bal - STARTING_BALANCE,
        "trades": len(trades), "wr": wins / len(trades) if trades else 0,
        "profit_factor": sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 1e-10),
        "avg_win": np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
        "avg_loss": np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
        "max_dd": 0,  # simplified
        "total_fees": sum(t["pnl"] for t in trades) * 0,  # skip for speed
        "sl_pct": sum(1 for t in trades if t["reason"] == "SL") / len(trades) if trades else 0,
        "tp_pct": sum(1 for t in trades if t["reason"] == "TP") / len(trades) if trades else 0,
    }


print("\n=== COLLECTING DATA ===\n")
data, sigs, atrs = prepare_data()

print("\n=== STRATEGY IMPROVEMENT TESTS ===\n")

tests = [
    ("BASE: 2x ATR, 4:1 RR", {"rr": 4.0, "atr_mult": 2.0}),
    ("A: 3x ATR stop",                     {"rr": 4.0, "atr_mult": 3.0}),
    ("B: 2x ATR + breakeven at 12h",        {"rr": 4.0, "atr_mult": 2.0, "breakeven_bars": 12}),
    ("C: 3x ATR + breakeven at 12h",        {"rr": 4.0, "atr_mult": 3.0, "breakeven_bars": 12}),
    ("D: Short 3:1 / Long 4:1",             {"rr": 4.0, "atr_mult": 2.0, "short_rr": 3.0}),
    ("E: No SL first 3 bars",               {"rr": 4.0, "atr_mult": 2.0, "grace_bars": 3}),
    ("F: 3x ATR + breakeven 12h + short 3:1", {"rr": 4.0, "atr_mult": 3.0, "breakeven_bars": 12, "short_rr": 3.0}),
    ("G: 4x ATR + breakeven 12h",            {"rr": 4.0, "atr_mult": 4.0, "breakeven_bars": 12}),
]

for label, kwargs in tests:
    r = backtest(data, sigs, atrs, **kwargs)
    print(f"\n  {label}")
    print(f"    Final: ${r['final']:.2f} | PnL: ${r['pnl']:+.2f} | Trades: {r['trades']} | WR: {r['wr']:.0%}")
    print(f"    Profit factor: {r['profit_factor']:.2f} | Avg win: ${r['avg_win']:.2f} | Avg loss: ${r['avg_loss']:.2f}")
    print(f"    SL: {r['sl_pct']:.0%} | TP: {r['tp_pct']:.0%}")
