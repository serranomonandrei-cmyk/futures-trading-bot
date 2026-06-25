"""
Multi-coin backtest. Exact same logic as bot_multi.py (single source of truth).
Imports STRATEGIES, calc_signal, calc_atr, and all params from bot_multi.
"""
import pandas as pd, numpy as np, ccxt
from data import fetch_ohlcv

exchange = ccxt.binance({"enableRateLimit": True})

import bot_multi
from bot_multi import STRATEGIES, calc_signal, calc_atr, LEVERAGE, RISK_PCT, RR_RATIO, ATR_STOP_MULT, MAX_BARS_HELD, MAX_POSITIONS, BREAKEVEN_BARS
from config import TAKER_FEE_PCT, SLIPPAGE_PCT, MIN_STOP_DISTANCE_PCT, MAX_MARGIN_UTILIZATION, STARTING_BALANCE

def run(min_months=12):
    print(f"\nMulti-coin backtest ({len(STRATEGIES)} coins, {MAX_POSITIONS} max pos)")
    print(f"Params: {LEVERAGE}x lev, {RISK_PCT:.0%} risk, {RR_RATIO}:1 RR, ATR {ATR_STOP_MULT}x stop, {MAX_BARS_HELD}h cap, breakeven at {BREAKEVEN_BARS}h")
    print()

    # Load data
    data = {}
    for pair in STRATEGIES:
        coin = pair.split("/")[0]
        df = fetch_ohlcv(exchange, pair, "1h", min_months)
        data[pair] = df
        print(f"  {coin}: {len(df)} bars")

    # Precompute signals (bar-by-bar, matching calc_signal in bot_multi)
    sigs = {}
    atrs = {}
    for pair, df in data.items():
        sigs[pair] = pd.Series(0, index=df.index)
        for i in range(len(df)):
            sigs[pair].iloc[i] = calc_signal(df.iloc[:i+1], pair)
        atrs[pair] = calc_atr(df)

    # Shared account simulation (matches bot_multi._open_position + _check_position)
    bal = float(STARTING_BALANCE)
    peak = bal
    positions = {}
    trades = []

    master = data["ETH/USDT:USDT"]
    n = len(master)

    for i in range(1, n):
        ts = master["timestamp"].iloc[i]

        # --- CLOSE positions (matches bot_multi._check_position) ---
        for pair in list(positions.keys()):
            pdf = data[pair]
            row_mask = pdf["timestamp"] == ts
            if not row_mask.any(): continue
            row = pdf[row_mask].iloc[0]
            pos = positions[pair]
            pos["bars"] += 1

            # Breakeven: move stop to entry after BREAKEVEN_BARS
            stop = pos["entry"] if pos["bars"] >= BREAKEVEN_BARS else pos["stop"]

            exit_p = None; reason = None
            if pos["bars"] >= MAX_BARS_HELD:
                exit_p = row["close"]; reason = "TIME"
            elif pos["side"] == "long":
                if row["low"] <= stop: exit_p = stop; reason = "SL"
                elif row["high"] >= pos["tp"]: exit_p = pos["tp"]; reason = "TP"
            else:
                if row["high"] >= stop: exit_p = stop; reason = "SL"
                elif row["low"] <= pos["tp"]: exit_p = pos["tp"]; reason = "TP"

            if exit_p is not None:
                pnl_pct = (exit_p - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - exit_p) / pos["entry"] - SLIPPAGE_PCT
                gross = pnl_pct * (pos["margin"] * LEVERAGE)
                fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                bal += net
                trades.append({"pair": pair, "pnl": net, "reason": reason})
                del positions[pair]

        # --- OPEN positions (matches bot_multi._open_position) ---
        if len(positions) < MAX_POSITIONS and bal > 0:
            for pair in STRATEGIES:
                if pair in positions or len(positions) >= MAX_POSITIONS: continue
                pdf = data[pair]
                row_mask = pdf["timestamp"] == ts
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
                if total_margin + margin > bal * MAX_MARGIN_UTILIZATION:
                    continue

                positions[pair] = {"side": side, "entry": entry, "stop": stop, "tp": tp, "margin": margin, "bars": 0}
                peak = max(peak, bal)

        peak = max(peak, bal)

    # Close remaining at market (matches bot_multi behavior on restart)
    for pair in list(positions.keys()):
        last = data[pair].iloc[-1]
        pos = positions[pair]
        pnl_pct = (last["close"] - pos["entry"]) / pos["entry"] - SLIPPAGE_PCT if pos["side"] == "long" else (pos["entry"] - last["close"]) / pos["entry"] - SLIPPAGE_PCT
        gross = pnl_pct * (pos["margin"] * LEVERAGE)
        fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
        net = gross - fees
        bal += net
        trades.append({"pair": pair, "pnl": net, "reason": "EOD"})

    # Results
    wins = sum(1 for t in trades if t["pnl"] > 0)
    coin_pnl = {}
    for t in trades:
        c = t["pair"].split("/")[0]
        coin_pnl[c] = coin_pnl.get(c, 0) + t["pnl"]

    print(f"\n  Final: ${bal:.2f} | PnL: ${bal-STARTING_BALANCE:+.2f} | Trades: {len(trades)} | WR: {wins/len(trades)*100:.0f}%" if trades else "  No trades")
    for c, p in sorted(coin_pnl.items(), key=lambda x: -x[1]):
        print(f"    {c:6s}: ${p:+.2f}")

if __name__ == "__main__":
    run()
