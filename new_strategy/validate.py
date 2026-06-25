"""
SMA strategy validation — regime analysis + multi-coin walk-forward.
Tests robustness across bull/bear/sideways regimes and rolling OOS windows.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ccxt
import numpy as np
import pandas as pd
from data import fetch_ohlcv
from backtest import Backtester
from config import PAIRS, TAKER_FEE_PCT, SLIPPAGE_PCT, MIN_STOP_DISTANCE_PCT
from config import MAX_MARGIN_UTILIZATION, STARTING_BALANCE
from bot_multi import STRATEGIES, calc_signal, calc_atr as calc_atr_orig

LEVERAGE = 15
RISK_PCT = 0.03
RR_RATIO = 4.0
ATR_STOP_MULT = 2.0
MAX_BARS_HELD = 72
MAX_POSITIONS = 10


def calc_atr(df):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def classify_regime(df, lookback=50):
    """Classify each bar as bull/bear/sideways based on SMA(lookback)."""
    sma = df["close"].rolling(lookback).mean()
    slope = sma.diff(5)  # slope over 5 bars
    regime = pd.Series("sideways", index=df.index)
    regime[(df["close"] > sma * 1.02) & (slope > 0)] = "bull"
    regime[(df["close"] < sma * 0.98) & (slope < 0)] = "bear"
    return regime


def fast_multi_backtest(coin_dfs, pair_list, strategy_dict, months_label=""):
    """Run multi-coin backtest using pre-fetched data dict {coin: df}."""
    fund = STARTING_BALANCE
    peak = fund
    trades = []

    # Precompute signal series for every coin
    sig_series = {}
    for pair in pair_list:
        coin = pair.split("/")[0]
        df = coin_dfs[coin]
        sig_series[coin] = calc_signal_series(df, pair)
        sig = sig_series[coin]
        atr = calc_atr(df).values
        coin_data[coin] = {
            "open": df["open"].values, "high": df["high"].values,
            "low": df["low"].values, "close": df["close"].values,
            "sig": sig.values if isinstance(sig, pd.Series) else np.zeros(len(df)),
            "atr": atr, "n": len(df),
        }

    n_bars = max(c["n"] for c in coin_data.values())
    opens = []

    for i in range(1, n_bars):
        still_open = []
        for p in opens:
            coin = p["coin"]
            d = coin_data[coin]
            if i >= d["n"]: continue
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
                trades.append({"coin": coin, "pnl": net, "side": p["side"], "regime": p.get("regime", "?")})
            else:
                still_open.append(p)
        opens = still_open

        for coin, d in coin_data.items():
            if i >= d["n"]: continue
            if len(opens) >= MAX_POSITIONS: break
            sig = d["sig"][i-1]
            if sig == 0: continue
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

            opens.append({"coin": coin, "side": side, "entry": entry, "stop": stop, "tp": tp,
                          "margin": margin, "bars": 0, "regime": "?"})

    for p in opens:
        coin, d = p["coin"], coin_data[p["coin"]]
        last = d["close"][d["n"] - 1]
        ret = (last - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - last) / p["entry"]
        ret -= SLIPPAGE_PCT
        fund += ret * (p["margin"] * LEVERAGE) - p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    coin_pnl = {}
    for t in trades:
        coin_pnl[t["coin"]] = coin_pnl.get(t["coin"], 0) + t["pnl"]

    return {
        "final": fund, "pnl": fund - STARTING_BALANCE,
        "trades": len(trades), "wr": len(wins) / len(trades) if trades else 0,
        "coin_pnl": coin_pnl, "peak": peak,
    }


def sma_signal_simple(df, period):
    """Simplified SMA signal — same logic as bot_multi.calc_signal for SMA type."""
    s = df["close"].rolling(period).mean()
    cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
    cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cu] = 1; sig[cd] = -1
    return sig


def optimize_sma_period(df, periods=range(5, 61)):
    """Find best SMA period for a coin (full backtest)."""
    best_p, best_pnl = None, -999
    for p in periods:
        fn = lambda d, pp=p: sma_signal_simple(d, pp)
        bt = Backtester("?", f"SMA({p})", fn)
        _, m = bt.run(df.copy())
        if "error" in m: continue
        if m["net_pnl"] > best_pnl:
            best_pnl = m["net_pnl"]
            best_p = p
    return best_p, best_pnl


def run_multi_walkforward(coin_dfs, train_months=2, test_months=1):
    """Rolling walk-forward — precomputes all signals, tests on OOS windows."""
    n_total = min(len(v) for v in coin_dfs.values())
    test_bars = test_months * 30 * 24

    # Precompute full signal series + ATR for every coin
    all_sigs = {}
    all_atrs = {}
    for pair in PAIRS:
        coin = pair.split("/")[0]
        df = coin_dfs[coin]
        all_sigs[coin] = calc_signal_series(df, pair)
        all_atrs[coin] = calc_atr(df).values

    oos_results = []
    start = train_months * 30 * 24

    print(f"\n  Walk-forward: {test_months}m test windows, fixed SMA periods", flush=True)

    while start + test_bars <= n_total:
        test_end = start + test_bars

        bal = STARTING_BALANCE
        trades = []
        positions = {}

        for i in range(start + 1, test_end):
            if i >= n_total: break

            # Close positions
            for pair in list(positions.keys()):
                coin = pair.split("/")[0]
                pdf = coin_dfs[coin]
                if i >= len(pdf): continue
                hi, lo, cl = pdf["high"].iloc[i], pdf["low"].iloc[i], pdf["close"].iloc[i]
                pos = positions[pair]
                pos["bars"] += 1
                ex = None; reason = ""
                if pos["bars"] >= MAX_BARS_HELD: ex, reason = cl, "TIME"
                elif pos["side"] == "long":
                    if lo <= pos["stop"]: ex, reason = pos["stop"], "SL"
                    elif hi >= pos["tp"]: ex, reason = pos["tp"], "TP"
                else:
                    if hi >= pos["stop"]: ex, reason = pos["stop"], "SL"
                    elif lo <= pos["tp"]: ex, reason = pos["tp"], "TP"
                if ex is not None:
                    ret = (ex - pos["entry"]) / pos["entry"] if pos["side"] == "long" else (pos["entry"] - ex) / pos["entry"]
                    ret -= SLIPPAGE_PCT
                    gross = ret * (pos["margin"] * LEVERAGE)
                    fees = pos["margin"] * LEVERAGE * TAKER_FEE_PCT * 2
                    net = gross - fees
                    bal += net
                    trades.append({"coin": coin, "pnl": net})
                    del positions[pair]

            # Open positions
            if len(positions) < MAX_POSITIONS and bal > 0:
                for pair in PAIRS:
                    if pair in positions or len(positions) >= MAX_POSITIONS: continue
                    coin = pair.split("/")[0]
                    sig = all_sigs[coin].iloc[i-1] if i-1 < len(all_sigs[coin]) else 0
                    if sig == 0: continue
                    pdf = coin_dfs[coin]
                    if i >= len(pdf): continue
                    atr_v = all_atrs[coin][i] if i < len(all_atrs[coin]) else np.nan
                    if np.isnan(atr_v) or atr_v <= 0: continue
                    side = "long" if sig == 1 else "short"
                    entry = pdf["open"].iloc[i] * (1 + SLIPPAGE_PCT if sig == 1 else 1 - SLIPPAGE_PCT)
                    dist = max(atr_v * ATR_STOP_MULT, entry * MIN_STOP_DISTANCE_PCT)
                    stop = entry - dist if side == "long" else entry + dist
                    tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO
                    risk_usd = bal * RISK_PCT
                    stop_dist_pct = abs(entry - stop) / entry
                    if stop_dist_pct < 1e-6: continue
                    notional = min(risk_usd / stop_dist_pct, 500)
                    margin = notional / LEVERAGE
                    if margin < 1: continue
                    total_m = sum(p["margin"] for p in positions.values())
                    if total_m + margin > bal * MAX_MARGIN_UTILIZATION: continue
                    positions[pair] = {"side": side, "entry": entry, "stop": stop, "tp": tp, "margin": margin, "bars": 0}

        for pair in list(positions.keys()):
            coin = pair.split("/")[0]
            pdf = coin_dfs[coin]
            last = pdf["close"].iloc[min(test_end, len(pdf)) - 1]
            p = positions[pair]
            ret = (last - p["entry"]) / p["entry"] if p["side"] == "long" else (p["entry"] - last) / p["entry"]
            ret -= SLIPPAGE_PCT
            bal += ret * (p["margin"] * LEVERAGE) - p["margin"] * LEVERAGE * TAKER_FEE_PCT * 2

        oos_pnl = bal - STARTING_BALANCE
        oos_results.append({"window": f"{start}-{test_end}", "pnl": oos_pnl, "trades": len(trades)})
        print(f"    OOS {start}-{test_end}: PnL ${oos_pnl:+.2f} | {len(trades)} trades", flush=True)
        start += test_bars

    total_oos = sum(w["pnl"] for w in oos_results)
    prof_win = sum(1 for w in oos_results if w["pnl"] > 0)
    print(f"\n  Walk-forward total OOS: ${total_oos:+.2f} ({prof_win}/{len(oos_results)} windows)")
    return oos_results


# ============================================================
def calc_signal_series(df, pair):
    """Full signal Series (not just last value)."""
    strat = STRATEGIES.get(pair)
    if not strat: return pd.Series(0, index=df.index)
    if strat["type"] == "SMA":
        s = df["close"].rolling(strat["period"]).mean()
        cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
        cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    elif strat["type"] == "EMA":
        s = df["close"].ewm(span=strat["period"], adjust=False).mean()
        cu = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
        cd = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    else:
        return pd.Series(0, index=df.index)
    sig = pd.Series(0, index=df.index)
    sig[cu] = 1; sig[cd] = -1
    return sig

# Fetch data for all coins
exchange = ccxt.binance({"enableRateLimit": True})
coin_dfs = {}
print("\n=== FETCHING DATA ===\n")
for pair in PAIRS:
    coin = pair.split("/")[0]
    print(f"  {coin}...", flush=True)
    coin_dfs[coin] = fetch_ohlcv(exchange, pair, months=6)
    print(f"    {len(coin_dfs[coin])} bars")

# ============================================================
# PART 1: Regime Analysis
print("\n" + "=" * 72)
print("  REGIME ANALYSIS — SMA per-coin optimal strategies")
print("=" * 72)

all_trades_by_regime = {"bull": [], "bear": [], "sideways": []}

for pair in PAIRS:
    coin = pair.split("/")[0]
    df = coin_dfs[coin]
    regime = classify_regime(df)
    sig = calc_signal_series(df, pair)
    atr = calc_atr(df).values
    bal = STARTING_BALANCE

    # Simulate trades, tracking regime at entry
    for i in range(1, len(df)):
        sig_val = sig.iloc[i-1] if isinstance(sig, pd.Series) else sig[i-1]
        if sig_val == 0: continue
        atr_v = atr[i]
        if np.isnan(atr_v) or atr_v <= 0: continue

        side = "long" if sig_val == 1 else "short"
        entry = df["open"].iloc[i] * (1 + SLIPPAGE_PCT if sig_val == 1 else 1 - SLIPPAGE_PCT)
        dist = max(atr_v * ATR_STOP_MULT, entry * MIN_STOP_DISTANCE_PCT)
        stop = entry - dist if side == "long" else entry + dist
        tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO

        r = regime.iloc[i] if isinstance(regime, pd.Series) else "sideways"

        risk_usd = bal * RISK_PCT
        stop_dist_pct = abs(entry - stop) / entry
        if stop_dist_pct < 1e-6: continue
        notional = min(risk_usd / stop_dist_pct, 500)
        margin = notional / LEVERAGE
        if margin < 1: continue
        bal -= margin

        # Find exit: scan forward bars
        for j in range(i, min(i + MAX_BARS_HELD, len(df))):
            hi, lo, cl = df["high"].iloc[j], df["low"].iloc[j], df["close"].iloc[j]
            hit = False; ex = 0.0
            if j - i >= MAX_BARS_HELD:
                ex, hit = cl, True
            elif side == "long":
                if lo <= stop: ex, hit = stop, True
                elif hi >= tp: ex, hit = tp, True
            else:
                if hi >= stop: ex, hit = stop, True
                elif lo <= tp: ex, hit = tp, True
            if hit:
                ret = (ex - entry) / entry if side == "long" else (entry - ex) / entry
                ret -= SLIPPAGE_PCT
                gross = ret * (margin * LEVERAGE)
                fees = margin * LEVERAGE * TAKER_FEE_PCT * 2
                net = gross - fees
                all_trades_by_regime[r].append({"coin": coin, "pnl": net, "side": side, "bars": j - i})
                bal += margin + net
                break

# Report regime performance
print("\n  Per-Regime PnL:")
for reg in ["bull", "bear", "sideways"]:
    ts = all_trades_by_regime[reg]
    if not ts:
        print(f"    {reg:10s}: 0 trades")
        continue
    pnls = [t["pnl"] for t in ts]
    wins = [p for p in pnls if p > 0]
    print(f"    {reg:10s}: PnL ${sum(pnls):>+8.2f} | WR {len(wins)/len(pnls):.0%} | Trades {len(pnls)}")

# ============================================================
# PART 2: Multi-coin walk-forward
print("\n" + "=" * 72)
print("  MULTI-COIN WALK-FORWARD")
print("=" * 72)

wf = run_multi_walkforward(coin_dfs, train_months=2, test_months=1)

# ============================================================
# PART 3: Full multi-coin backtest with original strategies
print("\n" + "=" * 72)
print("  FULL MULTI-COIN BACKTEST (original SMA strategies)")
print("=" * 72)

result = fast_multi_backtest(coin_dfs, PAIRS, STRATEGIES, "6mo")
coin_sorted = sorted(result["coin_pnl"].items(), key=lambda x: x[1], reverse=True)
print(f"\n  Final: ${result['final']:.2f} | PnL: ${result['pnl']:+.2f} | "
      f"Trades: {result['trades']} | WR: {result['wr']:.0%}")
for c, p in coin_sorted:
    arrow = "+" if p > 0 else ""
    print(f"    {c:6s}: ${arrow}{p:>8.2f}")

# ============================================================
# Summary
print("\n" + "=" * 72)
print("  VALIDATION SUMMARY")
print("=" * 72)

regime_summary = {}
total_pnl_by_reg = {}
for reg, ts in all_trades_by_regime.items():
    pnl = sum(t["pnl"] for t in ts) if ts else 0
    total_pnl_by_reg[reg] = pnl
print(f"  Regime: bull ${total_pnl_by_reg.get('bull', 0):+.2f}, "
      f"bear ${total_pnl_by_reg.get('bear', 0):+.2f}, "
      f"sideways ${total_pnl_by_reg.get('sideways', 0):+.2f}")

wf_total = sum(w["pnl"] for w in wf) if wf else 0
wf_wins = sum(1 for w in wf if w["pnl"] > 0) if wf else 0
print(f"  Walk-forward: ${wf_total:+.2f} ({wf_wins}/{len(wf)} windows)")
print(f"  IS full port: ${result['pnl']:+.2f}")

out = os.path.join(os.path.dirname(__file__), "validation.json")
with open(out, "w") as f:
    json.dump({
        "regime": {r: {"pnl": sum(t["pnl"] for t in ts), "trades": len(ts)}
                   for r, ts in all_trades_by_regime.items()},
        "walkforward": wf,
        "full_portfolio": result,
    }, f, indent=2, default=str)
print(f"\n  Results saved to {out}")
