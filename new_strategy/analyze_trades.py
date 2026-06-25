"""
Trade-level analysis. Runs backtest, captures every trade, finds patterns.
Answers: what makes a trade win/lose? How to improve?
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

LEVERAGE = 15; RISK_PCT = 0.03; RR_RATIO = 4.0
ATR_STOP_MULT = 2.0; MAX_BARS_HELD = 72; MAX_POSITIONS = 10


def collect_trades():
    """Run backtest, collect trade-level data for analysis."""
    exchange = ccxt.binance({"enableRateLimit": True})

    data = {}
    for pair in STRATEGIES:
        coin = pair.split("/")[0]
        data[pair] = fetch_ohlcv(exchange, pair, months=6)
        print(f"  {coin}: {len(data[pair])} bars")

    # Precompute signals and ATR
    sigs, atrs = {}, {}
    for pair, df in data.items():
        s = pd.Series(0, index=df.index)
        for i in range(len(df)):
            s.iloc[i] = calc_signal(df.iloc[:i+1], pair)
        sigs[pair] = s
        atrs[pair] = calc_atr(df)

    # Run shared account simulation, capture detailed trades
    bal = float(STARTING_BALANCE)
    peak = bal
    positions = {}
    trades = []
    master = data["ETH/USDT:USDT"]
    n = len(master)

    for i in range(1, n):
        ts = master["timestamp"].iloc[i]

        # Close
        for pair in list(positions.keys()):
            pdf = data[pair]; row_mask = pdf["timestamp"] == ts
            if not row_mask.any(): continue
            row = pdf[row_mask].iloc[0]
            pos = positions[pair]; pos["bars"] += 1
            exit_p = None; reason = None
            if pos["bars"] >= MAX_BARS_HELD:
                exit_p = row["close"]; reason = "TIME"
            elif pos["side"] == "long":
                if row["low"] <= pos["stop"]: exit_p = pos["stop"]; reason = "SL"
                elif row["high"] >= pos["tp"]: exit_p = pos["tp"]; reason = "TP"
            else:
                if row["high"] >= pos["stop"]: exit_p = pos["stop"]; reason = "SL"
                elif row["low"] <= pos["tp"]: exit_p = pos["tp"]; reason = "TP"
            if exit_p is not None:
                pnl_pct = (exit_p - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - exit_p) / pos["entry"] - SLIPPAGE_PCT
                gross = pnl_pct * (pos["margin"] * LEVERAGE)
                fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                bal += net
                trades.append({
                    "coin": pair.split("/")[0], "side": pos["side"],
                    "entry_price": pos["entry"], "exit_price": exit_p,
                    "entry_idx": pos["entry_idx"], "exit_idx": i,
                    "bars": pos["bars"], "reason": reason,
                    "atr_entry": pos["atr_entry"], "margin": pos["margin"],
                    "pnl": net, "pnl_pct": pnl_pct,
                    "stop_dist": abs(pos["entry"] - pos["stop"]),
                    "tp_dist": abs(pos["tp"] - pos["entry"]),
                    "entry_balance": pos["entry_balance"],
                    "balance_after": bal,
                    "stop": pos["stop"], "tp": pos["tp"],
                })
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
                dist = max(atr_val * ATR_STOP_MULT, entry * MIN_STOP_DISTANCE_PCT)
                stop = entry - dist if side == "long" else entry + dist
                tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO
                risk_usd = bal * RISK_PCT
                stop_dist_pct = abs(entry - stop) / entry
                if stop_dist_pct < 1e-6: continue
                notional = min(risk_usd / stop_dist_pct, 500)
                margin = notional / LEVERAGE
                if margin < 1.0: continue
                total_margin = sum(p["margin"] for p in positions.values())
                if total_margin + margin > bal * MAX_MARGIN_UTILIZATION: continue
                positions[pair] = {"side": side, "entry": entry, "stop": stop, "tp": tp,
                                   "margin": margin, "bars": 0, "entry_idx": i,
                                   "atr_entry": atr_val, "entry_balance": bal}
                peak = max(peak, bal)

    # Close remaining
    for pair in list(positions.keys()):
        last = data[pair].iloc[-1]; pos = positions[pair]
        pnl_pct = (last["close"] - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - last["close"]) / pos["entry"] - SLIPPAGE_PCT
        gross = pnl_pct * (pos["margin"] * LEVERAGE)
        fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
        net = gross - fees
        bal += net
        trades.append({
            "coin": pair.split("/")[0], "side": pos["side"],
            "entry_price": pos["entry"], "exit_price": last["close"],
            "entry_idx": pos["entry_idx"], "exit_idx": n,
            "bars": pos["bars"], "reason": "FORCE",
            "atr_entry": pos["atr_entry"], "margin": pos["margin"],
            "pnl": net, "pnl_pct": pnl_pct,
            "entry_balance": pos["entry_balance"], "balance_after": bal,
            "stop": pos["stop"], "tp": pos["tp"],
        })

    return trades, bal


def analyze(trades, final_bal):
    print("\n" + "=" * 72)
    print("  TRADE ANALYSIS")
    print("=" * 72)

    df = pd.DataFrame(trades)
    print(f"\n  Total trades: {len(df)}")
    print(f"  Final balance: ${final_bal:.2f} | PnL: ${final_bal-STARTING_BALANCE:+.2f}")

    # 1. Win/Loss by exit reason
    print("\n  --- BY EXIT REASON ---")
    for r in ["SL", "TP", "TIME", "FORCE"]:
        sub = df[df["reason"] == r]
        if len(sub) == 0: continue
        wins = (sub["pnl"] > 0).sum()
        print(f"    {r:6s}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f} | Avg ${sub['pnl'].mean():+>+8.2f}")

    # 2. By side
    print("\n  --- BY SIDE ---")
    for s in ["long", "short"]:
        sub = df[df["side"] == s]
        wins = (sub["pnl"] > 0).sum()
        print(f"    {s:6s}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f} | Avg ${sub['pnl'].mean():+>+8.2f}")

    # 3. By coin
    print("\n  --- BY COIN ---")
    for c in sorted(df["coin"].unique()):
        sub = df[df["coin"] == c]
        wins = (sub["pnl"] > 0).sum()
        print(f"    {c:6s}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f} | Avg ${sub['pnl'].mean():+>+8.2f}")

    # 4. By holding period buckets
    print("\n  --- BY HOLDING PERIOD ---")
    df["bar_bucket"] = pd.cut(df["bars"], bins=[0, 1, 3, 6, 12, 24, 48, 999], labels=["1", "2-3", "4-6", "7-12", "13-24", "25-48", "49+"])
    for label, sub in df.groupby("bar_bucket", observed=True):
        wins = (sub["pnl"] > 0).sum()
        print(f"    {label:>5s}h: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f} | Avg ${sub['pnl'].mean():+>+8.2f}")

    # 5. PnL distribution
    print("\n  --- PNL DISTRIBUTION ---")
    pnl = df["pnl"]
    print(f"    Avg win: ${pnl[pnl > 0].mean():+.4f}")
    print(f"    Avg loss: ${pnl[pnl < 0].mean():+.4f}")
    print(f"    Best trade: ${pnl.max():+.4f}")
    print(f"    Worst trade: ${pnl.min():+.4f}")
    print(f"    Median: ${pnl.median():+.4f}")
    print(f"    Std dev: ${pnl.std():.4f}")
    print(f"    Profit factor: {pnl[pnl>0].sum() / abs(pnl[pnl<0].sum()):.2f}")
    print(f"    Expectancy: ${pnl.mean():+.4f}")

    # 6. Consecutive losses
    print("\n  --- DRAWDOWN ANALYSIS ---")
    df_sorted = df.sort_values("exit_idx").reset_index(drop=True)
    max_consec_losses = 0; cur_losses = 0
    for _, r in df_sorted.iterrows():
        if r["pnl"] <= 0:
            cur_losses += 1
            max_consec_losses = max(max_consec_losses, cur_losses)
        else:
            cur_losses = 0
    print(f"    Max consecutive losses: {max_consec_losses}")

    # 7. ATR distance analysis
    print("\n  --- ATR AT ENTRY ---")
    df["atr_pct"] = df["atr_entry"] / df["entry_price"] * 100
    atr_bins = pd.cut(df["atr_pct"], bins=5)
    for label, sub in df.groupby(atr_bins, observed=True):
        wins = (sub["pnl"] > 0).sum()
        print(f"    ATR {label}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f}")

    # 8. Entry balance (position sizing effect)
    print("\n  --- BY ENTRY BALANCE (position size) ---")
    df["bal_bucket"] = pd.qcut(df["entry_balance"], q=4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
    for label, sub in df.groupby("bal_bucket", observed=True):
        wins = (sub["pnl"] > 0).sum()
        print(f"    {label:>10s}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | Avg PnL ${sub['pnl'].mean():+>+7.2f} | Avg margin ${sub['margin'].mean():.2f}")

    # 9. Entry index (early vs late in backtest)
    print("\n  --- EARLY vs LATE (data period half) ---")
    mid = df["entry_idx"].median()
    for label, sub in df.groupby(df["entry_idx"] < mid):
        wins = (sub["pnl"] > 0).sum()
        period = "First half" if label else "Second half"
        print(f"    {period:12s}: {len(sub):4d} trades | WR {wins/len(sub):.0%} | PnL ${sub['pnl'].sum():+>+8.2f} | Avg ${sub['pnl'].mean():+>+8.2f}")

    # 10. Monthly PnL
    print("\n  --- MONTHLY PNL ---")
    monthly = []
    for t in trades:
        idx = t["entry_idx"]
        month = idx // (30 * 24)  # approximate month from bar index
        monthly.append({"month": month, "pnl": t["pnl"]})
    mdf = pd.DataFrame(monthly)
    for month, sub in mdf.groupby("month"):
        print(f"    Month {int(month)+1:2d}: ${sub['pnl'].sum():+>+8.2f} ({len(sub)} trades)")

    # 11. Signal strength analysis
    print("\n  --- EXIT REASON BY SIDE ---")
    for s in ["long", "short"]:
        sub = df[df["side"] == s]
        sl = len(sub[sub["reason"] == "SL"])
        tp = len(sub[sub["reason"] == "TP"])
        tm = len(sub[sub["reason"] == "TIME"])
        print(f"    {s:6s}: SL {sl:3d} | TP {tp:3d} | TIME {tm:3d}")

    return df


if __name__ == "__main__":
    print("\n=== COLLECTING TRADES ===\n")
    trades, final = collect_trades()
    df = analyze(trades, final)
    # Save full trade data
    path = os.path.join(os.path.dirname(__file__), "trade_analysis.json")
    df.to_json(path, orient="records", indent=2)
    print(f"\n  Full trade data saved to {path}")
