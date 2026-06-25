"""
Run all 20 strategies on BTC/ETH/BNB. Fetch data, detect regimes, backtest, report.
"""

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

from config import PAIRS, BACKTEST_MONTHS, STARTING_BALANCE
from data import get_exchange, fetch_ohlcv, resample_ohlcv
from regime import detect_regime
from strategies import ALL_STRATEGIES
from backtest import Backtester


def fetch_data(exchange, pair, months):
    """Fetch 1H OHLCV and detect regime."""
    print(f"  Fetching {pair}...")
    df = fetch_ohlcv(exchange, pair, "1h", months)
    df = detect_regime(df)
    print(f"    {len(df)} bars, {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")
    return df


def run_all():
    exchange = get_exchange()
    all_results = []

    print(f"Starting balance: ${STARTING_BALANCE}")
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Strategies: {len(ALL_STRATEGIES)}")
    print(f"Period: {BACKTEST_MONTHS} months\n")

    # Fetch data for all pairs
    data = {}
    for pair in PAIRS:
        data[pair] = fetch_data(exchange, pair, BACKTEST_MONTHS)

    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS")
    print(f"{'='*80}\n")

    total = len(PAIRS) * len(ALL_STRATEGIES)
    done = 0

    for pair in PAIRS:
        df = data[pair]
        print(f"\n--- {pair} ---")

        for name, strategy_fn in ALL_STRATEGIES.items():
            done += 1
            bt = Backtester(pair, name, strategy_fn)
            try:
                portfolio, metrics = bt.run(df)
                metrics["pair"] = pair
                metrics["strategy"] = name
                all_results.append(metrics)

                # Quick status
                trades = metrics.get("total_trades", 0)
                wr = metrics.get("win_rate", 0)
                pnl = metrics.get("net_pnl", 0)
                dd = metrics.get("max_drawdown", 0)
                print(f"  [{done}/{total}] {name:25s} | {trades:3d} trades | WR {wr:.0%} | PnL ${pnl:+.2f} | DD {dd:.0%}")
            except Exception as e:
                print(f"  [{done}/{total}] {name:25s} | ERROR: {e}")
                all_results.append({"pair": pair, "strategy": name, "error": str(e)})

    # === RESULTS ===
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(all_results)
    df_valid = df_results[df_results["total_trades"].notna() & (df_results["total_trades"] > 0)].copy()

    if df_valid.empty:
        print("No valid results.")
        return

    # Sort by net PnL
    df_valid = df_valid.sort_values("net_pnl", ascending=False)

    print(f"{'Strategy':25s} {'Pair':15s} {'Trades':>6s} {'WR':>5s} {'PnL':>8s} {'PF':>6s} {'MaxDD':>6s} {'Sharpe':>7s} {'Final$':>8s}")
    print("-" * 95)

    for _, row in df_valid.iterrows():
        print(f"{row['strategy']:25s} {row['pair']:15s} {row['total_trades']:6.0f} {row['win_rate']:5.0%} ${row['net_pnl']:+7.2f} {row['profit_factor']:6.2f} {row['max_drawdown']:6.0%} {row['sharpe']:7.2f} ${row['final_balance']:7.2f}")

    # Top 5 per pair
    print(f"\n{'='*80}")
    print("TOP 5 STRATEGIES PER PAIR")
    print(f"{'='*80}")

    for pair in PAIRS:
        df_pair = df_valid[df_valid["pair"] == pair].head(5)
        if df_pair.empty:
            continue
        print(f"\n  {pair}:")
        for _, row in df_pair.iterrows():
            print(f"    {row['strategy']:25s} | {row['total_trades']:3.0f} trades | WR {row['win_rate']:.0%} | PnL ${row['net_pnl']:+.2f} | PF {row['profit_factor']:.2f}")

    # Overall top 10
    print(f"\n{'='*80}")
    print("OVERALL TOP 10 (by net PnL)")
    print(f"{'='*80}\n")

    top10 = df_valid.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. {row['strategy']:25s} on {row['pair']:15s} | PnL ${row['net_pnl']:+.2f} | WR {row['win_rate']:.0%} | DD {row['max_drawdown']:.0%}")

    # Save results
    df_valid.to_csv("/tmp/backtest_results.csv", index=False)
    print(f"\nFull results saved to /tmp/backtest_results.csv")

    # Summary stats
    print(f"\n{'='*80}")
    print("PORTFOLIO SUMMARY (if diversified across top strategies)")
    print(f"{'='*80}")

    # Simulate: start with $20, allocate $20/3 across 3 best pairs
    best_per_pair = {}
    for pair in PAIRS:
        df_pair = df_valid[df_valid["pair"] == pair]
        if not df_pair.empty:
            best_per_pair[pair] = df_pair.iloc[0]

    if best_per_pair:
        total_final = 0
        for pair, row in best_per_pair.items():
            alloc = STARTING_BALANCE / len(best_per_pair)
            final = alloc * (1 + row["return_pct"] / 100)
            total_final += final
            print(f"  {pair}: ${alloc:.2f} -> ${final:.2f} ({row['return_pct']:+.1f}%) via {row['strategy']}")
        print(f"\n  Total: ${STARTING_BALANCE:.2f} -> ${total_final:.2f} ({(total_final/STARTING_BALANCE-1)*100:+.1f}%)")


if __name__ == "__main__":
    run_all()
