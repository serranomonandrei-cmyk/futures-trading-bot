"""
Per-regime strategy search. Tests each strategy variant on its regime
using IS/OOS splits. Finds strategies profitable in BOTH periods.

Exit criteria: bull, bear, range each have at least 1 strategy
that is profitable in IS AND OOS individually.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
from three_strategies import BULL_STRATEGIES, BEAR_STRATEGIES, RANGE_STRATEGIES

exchange = get_exchange()
pair = "ETH/USDT:USDT"
df_all = fetch_ohlcv(exchange, pair, "1h", 12)
df_all = detect_regime(df_all)

# Split IS/OOS: 70% / 30%
split_date = pd.Timestamp("2026-02-23", tz="UTC")

# Use default ATR stop map (2.0x for trending, 1.5x for ranging)
default_atr = {
    "trending_up": 2.0, "trending_down": 2.0,
    "ranging": 1.5, "volatile": 2.5, "unknown": 2.5
}

total_is_bars = (df_all["timestamp"] < split_date).sum()
total_oos_bars = (df_all["timestamp"] >= split_date).sum()

print("=" * 70)
print(f"  PER-REGIME STRATEGY SEARCH")
print(f"  IS:  {df_all['timestamp'].min()} to {split_date} ({total_is_bars} bars)")
print(f"  OOS: {split_date} to {df_all['timestamp'].max()} ({total_oos_bars} bars)")
print("=" * 70)

# Regime distribution
for r in ["trending_up", "trending_down", "ranging"]:
    mask = df_all["regime"] == r
    is_mask = mask & (df_all["timestamp"] < split_date)
    oos_mask = mask & (df_all["timestamp"] >= split_date)
    print(f"  {r}: {mask.sum()} bars (IS: {is_mask.sum()}, OOS: {oos_mask.sum()})")

best_bull = None
best_bear = None
best_range = None

# ============================================================
# BULL SEARCH
# ============================================================
print(f"\n{'='*70}")
print("  BULL REGIME (trending_up) — STRATEGY SEARCH")
print("="*70)

bull_mask = df_all["regime"] == "trending_up"
df_bull_is = df_all[bull_mask & (df_all["timestamp"] < split_date)].copy().reset_index(drop=True)
df_bull_oos = df_all[bull_mask & (df_all["timestamp"] >= split_date)].copy().reset_index(drop=True)

print(f"  IS bars: {len(df_bull_is)} | OOS bars: {len(df_bull_oos)}")

for name, fn in BULL_STRATEGIES.items():
    bt = Backtester(pair, name, fn, atr_stop_map=default_atr)

    try:
        _, mis = bt.run(df_bull_is)
        is_pnl = mis.get("net_pnl", 0)
        is_t = mis.get("total_trades", 0)
        is_wr = mis.get("win_rate", 0)
        is_dd = mis.get("max_drawdown", 0)
    except Exception as e:
        is_pnl, is_t, is_wr, is_dd = 0, 0, 0, 0
        print(f"  {name:20s} IS ERROR: {e}")
        continue

    try:
        _, moos = bt.run(df_bull_oos)
        oos_pnl = moos.get("net_pnl", 0)
        oos_t = moos.get("total_trades", 0)
        oos_wr = moos.get("win_rate", 0)
        oos_dd = moos.get("max_drawdown", 0)
    except Exception as e:
        oos_pnl, oos_t, oos_wr, oos_dd = 0, 0, 0, 0

    is_ok = is_pnl > 0
    oos_ok = oos_pnl > 0
    total_ok = is_ok and oos_ok
    status = "GOLDEN" if total_ok else ("IS_OK" if is_ok else ("OOS_OK" if oos_ok else "FAIL"))
    mark = "***" if total_ok else ("+" if is_ok else ("~" if oos_ok else "-"))

    print(f"  {mark} {name:20s} | IS: {is_t:3d}t {is_wr:.0%} ${is_pnl:+7.2f} DD{is_dd:.0%} | OOS: {oos_t:3d}t {oos_wr:.0%} ${oos_pnl:+7.2f} DD{oos_dd:.0%} | {status}")

    if total_ok:
        total = is_pnl + oos_pnl
        if best_bull is None or total > (best_bull[0] + best_bull[2]):
            best_bull = (is_pnl, oos_pnl, total, name)


# ============================================================
# BEAR SEARCH
# ============================================================
print(f"\n{'='*70}")
print("  BEAR REGIME (trending_down) — STRATEGY SEARCH")
print("="*70)

bear_mask = df_all["regime"] == "trending_down"
df_bear_is = df_all[bear_mask & (df_all["timestamp"] < split_date)].copy().reset_index(drop=True)
df_bear_oos = df_all[bear_mask & (df_all["timestamp"] >= split_date)].copy().reset_index(drop=True)

print(f"  IS bars: {len(df_bear_is)} | OOS bars: {len(df_bear_oos)}")

for name, fn in BEAR_STRATEGIES.items():
    bt = Backtester(pair, name, fn, atr_stop_map=default_atr)

    try:
        _, mis = bt.run(df_bear_is)
        is_pnl = mis.get("net_pnl", 0)
        is_t = mis.get("total_trades", 0)
        is_wr = mis.get("win_rate", 0)
        is_dd = mis.get("max_drawdown", 0)
    except Exception as e:
        is_pnl, is_t, is_wr, is_dd = 0, 0, 0, 0
        print(f"  {name:20s} IS ERROR: {e}")
        continue

    try:
        _, moos = bt.run(df_bear_oos)
        oos_pnl = moos.get("net_pnl", 0)
        oos_t = moos.get("total_trades", 0)
        oos_wr = moos.get("win_rate", 0)
        oos_dd = moos.get("max_drawdown", 0)
    except Exception as e:
        oos_pnl, oos_t, oos_wr, oos_dd = 0, 0, 0, 0

    is_ok = is_pnl > 0
    oos_ok = oos_pnl > 0
    total_ok = is_ok and oos_ok
    status = "GOLDEN" if total_ok else ("IS_OK" if is_ok else ("OOS_OK" if oos_ok else "FAIL"))
    mark = "***" if total_ok else ("+" if is_ok else ("~" if oos_ok else "-"))

    print(f"  {mark} {name:20s} | IS: {is_t:3d}t {is_wr:.0%} ${is_pnl:+7.2f} DD{is_dd:.0%} | OOS: {oos_t:3d}t {oos_wr:.0%} ${oos_pnl:+7.2f} DD{oos_dd:.0%} | {status}")

    if total_ok:
        total = is_pnl + oos_pnl
        if best_bear is None or total > (best_bear[0] + best_bear[2]):
            best_bear = (is_pnl, oos_pnl, total, name)


# ============================================================
# RANGE SEARCH
# ============================================================
print(f"\n{'='*70}")
print("  RANGE REGIME (ranging) — STRATEGY SEARCH")
print("="*70)

range_mask = df_all["regime"] == "ranging"
df_range_is = df_all[range_mask & (df_all["timestamp"] < split_date)].copy().reset_index(drop=True)
df_range_oos = df_all[range_mask & (df_all["timestamp"] >= split_date)].copy().reset_index(drop=True)

print(f"  IS bars: {len(df_range_is)} | OOS bars: {len(df_range_oos)}")

for name, fn in RANGE_STRATEGIES.items():
    bt = Backtester(pair, name, fn, atr_stop_map=default_atr)

    try:
        _, mis = bt.run(df_range_is)
        is_pnl = mis.get("net_pnl", 0)
        is_t = mis.get("total_trades", 0)
        is_wr = mis.get("win_rate", 0)
        is_dd = mis.get("max_drawdown", 0)
    except Exception as e:
        is_pnl, is_t, is_wr, is_dd = 0, 0, 0, 0
        print(f"  {name:20s} IS ERROR: {e}")
        continue

    try:
        _, moos = bt.run(df_range_oos)
        oos_pnl = moos.get("net_pnl", 0)
        oos_t = moos.get("total_trades", 0)
        oos_wr = moos.get("win_rate", 0)
        oos_dd = moos.get("max_drawdown", 0)
    except Exception as e:
        oos_pnl, oos_t, oos_wr, oos_dd = 0, 0, 0, 0

    is_ok = is_pnl > 0
    oos_ok = oos_pnl > 0
    total_ok = is_ok and oos_ok
    status = "GOLDEN" if total_ok else ("IS_OK" if is_ok else ("OOS_OK" if oos_ok else "FAIL"))
    mark = "***" if total_ok else ("+" if is_ok else ("~" if oos_ok else "-"))

    print(f"  {mark} {name:20s} | IS: {is_t:3d}t {is_wr:.0%} ${is_pnl:+7.2f} DD{is_dd:.0%} | OOS: {oos_t:3d}t {oos_wr:.0%} ${oos_pnl:+7.2f} DD{oos_dd:.0%} | {status}")

    if total_ok:
        total = is_pnl + oos_pnl
        if best_range is None or total > (best_range[0] + best_range[2]):
            best_range = (is_pnl, oos_pnl, total, name)


# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*70}")
print("  FINAL RESULTS")
print("="*70)

print(f"\n  BULL: ", end="")
if best_bull:
    print(f"{best_bull[3]} — IS ${best_bull[0]:+.2f} OOS ${best_bull[1]:+.2f} Total ${best_bull[2]:+.2f}")
else:
    print("NO GOLDEN STRATEGY FOUND")

print(f"  BEAR: ", end="")
if best_bear:
    print(f"{best_bear[3]} — IS ${best_bear[0]:+.2f} OOS ${best_bear[1]:+.2f} Total ${best_bear[2]:+.2f}")
else:
    print("NO GOLDEN STRATEGY FOUND")

print(f"  RANGE: ", end="")
if best_range:
    print(f"{best_range[3]} — IS ${best_range[0]:+.2f} OOS ${best_range[1]:+.2f} Total ${best_range[2]:+.2f}")
else:
    print("NO GOLDEN STRATEGY FOUND")

all_golden = best_bull is not None and best_bear is not None and best_range is not None
print(f"\n  ALL THREE GOLDEN: {'YES!' if all_golden else 'NO'}")