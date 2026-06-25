"""
Systematic strategy search. Tests all archetypes × all parameter combos
on IS/OOS per regime. Finds the most profitable configurations.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
from three_strategies import ALL_STRATEGIES, STRAT_PARAMS
import backtest

# Optimized backtest settings
backtest.calc_tp = lambda s, e, stop, rr=4.0: e + abs(e-stop)*rr if s=="long" else e - abs(e-stop)*rr
atr_map = {"trending_up": 2.0, "trending_down": 2.0, "ranging": 1.5, "volatile": 2.5, "unknown": 2.5}
risk_map = {"trending_up": 0.03, "trending_down": 0.03, "ranging": 0.03, "volatile": 0.015, "unknown": 0.015}
lev_map = {"trending_up": 15, "trending_down": 15, "ranging": 10, "volatile": 5, "unknown": 5}

exchange = get_exchange()
pair = "ETH/USDT:USDT"
df_all = fetch_ohlcv(exchange, pair, "1h", 12)
df_all = detect_regime(df_all)
split = pd.Timestamp("2026-02-23", tz="UTC")

# Extract regime-specific data
regime_data = {}
for regime in ["trending_up", "trending_down", "ranging"]:
    mask = df_all["regime"] == regime
    df_is = df_all[mask & (df_all["timestamp"] < split)].copy().reset_index(drop=True)
    df_oos = df_all[mask & (df_all["timestamp"] >= split)].copy().reset_index(drop=True)
    regime_data[regime] = (df_is, df_oos)

print("=" * 80)
print("  SYSTEMATIC STRATEGY SEARCH — 8 ARCHETYPES × PARAMETER GRIDS")
print("  Target: IS > 0 AND OOS > 0 per regime")
print("=" * 80)

results = []

for strat_name, strat_fn in ALL_STRATEGIES.items():
    param_list = STRAT_PARAMS.get(strat_name, [{}])

    for pi, params in enumerate(param_list):
        param_label = f"{strat_name}_p{pi}"
        fn = lambda df, fn=strat_fn, p=params: fn(df, **p)

        # Test on each regime
        bull_is, bull_oos = 0, 0
        bear_is, bear_oos = 0, 0
        range_is, range_oos = 0, 0

        for regime in ["trending_up", "trending_down", "ranging"]:
            df_is, df_oos = regime_data[regime]
            if len(df_is) < 20 or len(df_oos) < 10:
                continue

            bt = Backtester(pair, param_label, fn, atr_stop_map=atr_map,
                           risk_map=risk_map, leverage_map=lev_map)

            try:
                _, mi = bt.run(df_is)
                _, mo = bt.run(df_oos)
                is_pnl = mi.get("net_pnl", 0)
                oos_pnl = mo.get("net_pnl", 0)
            except:
                is_pnl, oos_pnl = 0, 0

            if regime == "trending_up":
                bull_is, bull_oos = is_pnl, oos_pnl
            elif regime == "trending_down":
                bear_is, bear_oos = is_pnl, oos_pnl
            else:
                range_is, range_oos = is_pnl, oos_pnl

        total = bull_is + bull_oos + bear_is + bear_oos + range_is + range_oos
        golden_bull = bull_is > 0 and bull_oos > 0
        golden_bear = bear_is > 0 and bear_oos > 0
        golden_range = range_is > 0 and range_oos > 0
        golden_count = sum([golden_bull, golden_bear, golden_range])

        results.append({
            "name": param_label,
            "strategy": strat_name,
            "params": str(params),
            "bull_IS": bull_is, "bull_OOS": bull_oos,
            "bear_IS": bear_is, "bear_OOS": bear_oos,
            "range_IS": range_is, "range_OOS": range_oos,
            "total": total, "golden": golden_count,
        })

        if golden_count >= 1:
            stars = "*" * golden_count
            parts = []
            if golden_bull: parts.append(f"BULL ${bull_is:+.1f}/${bull_oos:+.1f}")
            if golden_bear: parts.append(f"BEAR ${bear_is:+.1f}/${bear_oos:+.1f}")
            if golden_range: parts.append(f"RANGE ${range_is:+.1f}/${range_oos:+.1f}")
            print(f"  {stars} {param_label:30s} | {' | '.join(parts)} | Total ${total:+.2f}")

# Sort by total profit
results.sort(key=lambda r: r["total"], reverse=True)
print(f"\n{'='*80}")
print("  TOP 10 BY TOTAL PROFIT")
print("="*80)

for r in results[:10]:
    g = "*" * r["golden"]
    print(f"  {g} {r['name']:30s} | Total ${r['total']:+8.2f} | "
          f"Bull ${r['bull_IS']:+6.1f}/${r['bull_OOS']:+6.1f} "
          f"Bear ${r['bear_IS']:+6.1f}/${r['bear_OOS']:+6.1f} "
          f"Range ${r['range_IS']:+6.1f}/${r['range_OOS']:+6.1f}")

# Best per regime
print(f"\n{'='*80}")
print("  BEST PER REGIME (by IS+OOS total)")
print("="*80)

for regime in ["bull", "bear", "range"]:
    key = f"{regime}_IS"
    best = max(results, key=lambda r: r[f"{regime}_IS"] + r[f"{regime}_OOS"])
    is_pnl = best[f"{regime}_IS"]
    oos_pnl = best[f"{regime}_OOS"]
    is_gold = is_pnl > 0 and oos_pnl > 0
    print(f"  {regime.upper()}: {best['name']} — IS ${is_pnl:+.2f} OOS ${oos_pnl:+.2f} Total ${is_pnl+oos_pnl:+.2f} {'GOLDEN' if is_gold else ''}")

# Summary stats
golden_strats = [r for r in results if r["golden"] >= 1]
print(f"\n  Total strategies tested: {len(results)}")
print(f"  Strategies with >=1 golden regime: {len(golden_strats)}")
if golden_strats:
    best_overall = max(golden_strats, key=lambda r: r["total"])
    print(f"  Best overall: {best_overall['name']} — Total ${best_overall['total']:+.2f}")