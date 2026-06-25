"""
Test regime-specific strategies on their specific periods.
Then combine into one bot and backtest full 12 months.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
from regime_strategies import (
    bull_pullback, bear_pullback, range_fade, range_swing, range_squeeze,
    BULL_STRATEGIES, BEAR_STRATEGIES, RANGE_STRATEGIES
)
from strategies import donchian_breakout, rsi_extreme
from range_variants import range_v9

exchange = get_exchange()

# Hardcoded regime periods (from data analysis)
PERIODS = {
    "ETH/USDT:USDT": {
        "trending_up": [
            ("2025-07-14", "2025-07-18", +11.2),
            ("2025-08-06", "2025-08-14", +18.4),
            ("2025-12-01", "2025-12-05", +12.2),
        ],
        "trending_down": [
            ("2026-01-19", "2026-01-22", -6.1),
            ("2026-01-30", "2026-02-03", -8.3),
            ("2026-02-26", "2026-03-01", -7.0),
            ("2026-06-04", "2026-06-07", -10.5),
        ],
        "ranging": [
            ("2025-07-01", "2025-07-13", 0),
            ("2025-09-01", "2025-11-30", 0),
            ("2026-03-01", "2026-05-31", 0),
        ],
    },
    "BNB/USDT:USDT": {
        "trending_up": [
            ("2025-10-11", "2025-10-14", +13.8),
            ("2025-12-01", "2025-12-04", +7.2),
            ("2026-05-29", "2026-06-01", +12.6),
        ],
        "trending_down": [
            ("2025-11-19", "2025-11-22", -6.9),
            ("2026-01-30", "2026-02-02", -2.6),
            ("2026-02-03", "2026-02-06", -17.8),
            ("2026-06-01", "2026-06-05", -11.7),
        ],
        "ranging": [
            ("2025-07-01", "2025-10-10", 0),
            ("2025-12-05", "2026-01-18", 0),
            ("2026-03-01", "2026-05-28", 0),
        ],
    },
    "BTC/USDT:USDT": {
        "trending_up": [],
        "trending_down": [
            ("2026-02-03", "2026-02-07", -13.3),
            ("2026-06-01", "2026-06-04", -6.1),
        ],
        "ranging": [
            ("2025-07-01", "2025-12-31", 0),
            ("2026-03-01", "2026-05-31", 0),
        ],
    },
}


def run_tests():
    all_results = []

    for pair, regimes in PERIODS.items():
        print(f"\n{'='*80}")
        print(f"  {pair}")
        print(f"{'='*80}")

        df_1h = fetch_ohlcv(exchange, pair, "1h", 12)
        df_1h = detect_regime(df_1h)

        for regime_type, periods in regimes.items():
            if not periods:
                print(f"\n  {regime_type.upper()}: No periods available")
                continue

            print(f"\n  --- {regime_type.upper()} ---")

            # Pick strategies for this regime
            if regime_type == "trending_up":
                strats = {**BULL_STRATEGIES, "donchian_20": lambda df: donchian_breakout(df, 20)}
            elif regime_type == "trending_down":
                strats = {**BEAR_STRATEGIES, "donchian_20": lambda df: donchian_breakout(df, 20)}
            else:
                strats = RANGE_STRATEGIES

            for strat_name, strat_fn in strats.items():
                total_pnl = 0
                total_trades = 0
                total_wins = 0
                max_dd = 0
                periods_tested = 0

                for start_str, end_str, expected in periods:
                    start = pd.Timestamp(start_str, tz="UTC")
                    end = pd.Timestamp(end_str, tz="UTC")
                    mask = (df_1h["timestamp"] >= start) & (df_1h["timestamp"] <= end)
                    df_period = df_1h[mask].copy().reset_index(drop=True)

                    if len(df_period) < 10:
                        continue

                    bt = Backtester(pair, strat_name, strat_fn)
                    try:
                        _, metrics = bt.run(df_period)
                        total_pnl += metrics.get("net_pnl", 0)
                        total_trades += metrics.get("total_trades", 0)
                        total_wins += metrics.get("wins", 0)
                        max_dd = max(max_dd, metrics.get("max_drawdown", 0))
                        periods_tested += 1
                    except:
                        pass

                wr = total_wins / total_trades if total_trades > 0 else 0
                flag = "OK" if total_pnl > 0 else "LOSS" if total_trades > 0 else "NO TRADES"
                print(f"    {strat_name:25s} | {total_trades:3d} trades | WR {wr:.0%} | PnL ${total_pnl:+6.2f} | DD {max_dd:.0%} | {flag}")

                all_results.append({
                    "pair": pair,
                    "regime": regime_type,
                    "strategy": strat_name,
                    "trades": total_trades,
                    "wins": total_wins,
                    "pnl": total_pnl,
                    "max_dd": max_dd,
                    "periods_tested": periods_tested,
                })

    # ============================================
    # COMBINED BACKTEST: Best strategy per regime
    # ============================================
    print(f"\n{'='*80}")
    print("COMBINED REGIME-ADAPTIVE BACKTEST (12 months)")
    print(f"{'='*80}\n")

    for pair in PERIODS:
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)

        # Strategy selection per regime
        if pair == "BTC/USDT:USDT":
            # BTC: only bear works, no bull periods
            strat_map = {
                "trending_up": lambda df: pd.Series(0, index=df.index),  # No signal
                "trending_down": lambda df: donchian_breakout(df, 20),
                "ranging": lambda df: pd.Series(0, index=df.index),  # No signal
            }
        else:
            # ETH/BNB: bull, bear, and ranging strategies
            strat_map = {
                "trending_up": lambda df: donchian_breakout(df, 20),
                "trending_down": lambda df: donchian_breakout(df, 20),
                "ranging": range_v9,  # Keltner fade (only profitable ranging strat)
            }

        def regime_strategy(df, _strat_map=strat_map):
            signals = pd.Series(0, index=df.index)
            for regime, fn in _strat_map.items():
                mask = df["regime"] == regime
                if mask.sum() < 10:
                    continue
                regime_signals = fn(df)
                signals[mask] = regime_signals[mask]
            return signals

        bt = Backtester(pair, "regime_adaptive", regime_strategy)
        portfolio, metrics = bt.run(df_full)

        t = metrics.get("total_trades", 0)
        wr = metrics.get("win_rate", 0)
        pnl = metrics.get("net_pnl", 0)
        dd = metrics.get("max_drawdown", 0)
        pf = metrics.get("profit_factor", 0)
        final = metrics.get("final_balance", 0)
        ret = metrics.get("return_pct", 0)
        print(f"  {pair}: {t} trades | WR {wr:.0%} | PnL ${pnl:+.2f} | PF {pf:.2f} | DD {dd:.0%} | {ret:+.1f}% | Final ${final:.2f}")

    # Also test: Donchian-only (baseline)
    print(f"\n  --- Baseline: Donchian-only ---")
    for pair in PERIODS:
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)
        bt = Backtester(pair, "donchian_only", lambda df: donchian_breakout(df, 20))
        _, metrics = bt.run(df_full)
        t = metrics.get("total_trades", 0)
        pnl = metrics.get("net_pnl", 0)
        dd = metrics.get("max_drawdown", 0)
        final = metrics.get("final_balance", 0)
        ret = metrics.get("return_pct", 0)
        print(f"  {pair}: {t} trades | PnL ${pnl:+.2f} | DD {dd:.0%} | {ret:+.1f}% | Final ${final:.2f}")

    # Save
    pd.DataFrame(all_results).to_csv("/tmp/regime_strat_results.csv", index=False)
    print(f"\nResults saved to /tmp/regime_strat_results.csv")


if __name__ == "__main__":
    run_tests()
