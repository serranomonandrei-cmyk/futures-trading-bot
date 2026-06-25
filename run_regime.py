"""
Regime-adaptive multi-strategy backtest.
Different strategy per regime. Hardcoded regime periods for validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data import get_exchange, fetch_ohlcv, resample_ohlcv
from regime import detect_regime
from strategies import (
    ema_crossover, triple_ema, donchian_breakout, adx_di, parabolic_sar,
    rsi_extreme, bollinger_bounce, stochastic_extreme,
    volume_breakout, atr_breakout, inside_bar_breakout, macd_crossover,
    ema_rsi_volume, hull_ma_crossover
)
from backtest import Backtester

# ============================================
# HARDCODED REGIME PERIODS (from data analysis)
# ============================================

REGIME_PERIODS = {
    "BTC/USDT:USDT": {
        "trending_up": [
            # BTC rarely trends up in this period
        ],
        "trending_down": [
            ("2026-02-03", "2026-02-07", -13.3),  # Feb crash
            ("2026-06-01", "2026-06-04", -6.1),    # Jun drop
        ],
        "ranging": [
            ("2025-07-01", "2025-09-30", 0),  # Q3 2025 range
            ("2025-10-01", "2025-12-31", 0),  # Q4 2025 range
            ("2026-01-01", "2026-02-02", 0),  # Jan range
            ("2026-03-01", "2026-06-01", 0),  # Mar-May range
        ],
    },
    "ETH/USDT:USDT": {
        "trending_up": [
            ("2025-07-14", "2025-07-18", 11.2),   # Jul pump
            ("2025-08-06", "2025-08-14", 18.4),   # Aug pump
            ("2025-12-01", "2025-12-05", 12.2),   # Dec pump
        ],
        "trending_down": [
            ("2026-01-19", "2026-01-22", -6.1),   # Jan drop
            ("2026-01-30", "2026-02-03", -8.3),   # Late Jan drop
            ("2026-02-26", "2026-03-01", -7.0),   # Feb drop
            ("2026-06-04", "2026-06-07", -10.5),  # Jun crash
        ],
        "ranging": [
            ("2025-07-01", "2025-07-13", 0),
            ("2025-07-19", "2025-11-30", 0),
            ("2025-12-06", "2026-01-18", 0),
            ("2026-03-01", "2026-06-01", 0),
        ],
    },
    "BNB/USDT:USDT": {
        "trending_up": [
            ("2025-10-11", "2025-10-14", 13.8),   # Oct pump
            ("2025-12-01", "2025-12-04", 7.2),    # Dec pump
            ("2026-05-29", "2026-06-01", 12.6),   # May pump
        ],
        "trending_down": [
            ("2025-11-19", "2025-11-22", -6.9),   # Nov drop
            ("2026-01-19", "2026-01-22", -3.9),   # Jan drop
            ("2026-01-30", "2026-02-02", -2.6),   # Late Jan
            ("2026-02-03", "2026-02-06", -17.8),  # Feb crash
            ("2026-06-01", "2026-06-05", -11.7),  # Jun crash
        ],
        "ranging": [
            ("2025-07-01", "2025-10-10", 0),
            ("2025-10-15", "2025-11-18", 0),
            ("2025-12-05", "2026-01-18", 0),
            ("2026-03-01", "2026-05-28", 0),
        ],
    },
}

# ============================================
# REGIME-SPECIFIC STRATEGIES
# ============================================

TREND_STRATEGIES = {
    "ema_cross_9_21": lambda df: ema_crossover(df, 9, 21),
    "triple_ema": triple_ema,
    "donchian_20": lambda df: donchian_breakout(df, 20),
    "adx_di": adx_di,
    "parabolic_sar": parabolic_sar,
    "ema_rsi_volume": ema_rsi_volume,
    "hull_ma": hull_ma_crossover,
    "macd_cross": macd_crossover,
}

RANGE_STRATEGIES = {
    "rsi_extreme": rsi_extreme,
    "bollinger_bounce": bollinger_bounce,
    "stochastic_extreme": stochastic_extreme,
    "ema_cross_5_13": lambda df: ema_crossover(df, 5, 13),
    "macd_cross": macd_crossover,
    "inside_bar": inside_bar_breakout,
}


def run_regime_tests():
    exchange = get_exchange()
    all_results = []

    for pair, regimes in REGIME_PERIODS.items():
        print(f"\n{'='*80}")
        print(f"  {pair} — REGIME-SPECIFIC TESTING")
        print(f"{'='*80}")

        # Fetch 12 months of data
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)

        for regime_type, periods in regimes.items():
            if not periods:
                continue

            print(f"\n  --- {regime_type.upper()} ---")

            for start_str, end_str, expected_move in periods:
                start = pd.Timestamp(start_str, tz="UTC")
                end = pd.Timestamp(end_str, tz="UTC")

                # Slice data to exact period
                mask = (df_full["timestamp"] >= start) & (df_full["timestamp"] <= end)
                df_period = df_full[mask].copy().reset_index(drop=True)

                if len(df_period) < 10:
                    continue

                # Test all strategies of this regime type
                strategies = TREND_STRATEGIES if "trend" in regime_type else RANGE_STRATEGIES

                for strat_name, strat_fn in strategies.items():
                    bt = Backtester(pair, strat_name, strat_fn)
                    try:
                        portfolio, metrics = bt.run(df_period)
                        metrics["pair"] = pair
                        metrics["strategy"] = strat_name
                        metrics["regime"] = regime_type
                        metrics["period"] = f"{start_str} to {end_str}"
                        metrics["expected_move"] = expected_move
                        metrics["period_bars"] = len(df_period)
                        all_results.append(metrics)

                        trades = metrics.get("total_trades", 0)
                        pnl = metrics.get("net_pnl", 0)
                        wr = metrics.get("win_rate", 0)
                        dd = metrics.get("max_drawdown", 0)
                        pf = metrics.get("profit_factor", 0)
                        flag = "OK" if pnl > 0 else "LOSS"
                        print(f"    {strat_name:25s} | {trades:3d} trades | WR {wr:.0%} | PnL ${pnl:+6.2f} | PF {pf:.2f} | DD {dd:.0%} | {flag}")
                    except Exception as e:
                        print(f"    {strat_name:25s} | ERROR: {e}")

    # ============================================
    # SUMMARY: Best strategy per regime
    # ============================================
    print(f"\n{'='*80}")
    print("BEST STRATEGY PER REGIME (sorted by PnL)")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)
    df_valid = df[df["total_trades"].notna() & (df["total_trades"] > 0)].copy()

    for regime in ["trending_up", "trending_down", "ranging"]:
        df_regime = df_valid[df_valid["regime"] == regime].sort_values("net_pnl", ascending=False)
        if df_regime.empty:
            continue
        print(f"  {regime.upper()}:")
        for _, row in df_regime.head(5).iterrows():
            print(f"    {row['strategy']:25s} {row['pair']:15s} | {row['period']:30s} | PnL ${row['net_pnl']:+6.2f} | WR {row['win_rate']:.0%} | PF {row['profit_factor']:.2f}")
        print()

    # ============================================
    # COMBINED REGIME-ADAPTIVE BACKTEST
    # ============================================
    print(f"{'='*80}")
    print("COMBINED REGIME-ADAPTIVE BACKTEST")
    print(f"{'='*80}\n")

    # Hardcoded: donchian for trending, rsi for ranging (from per-period results)
    def make_regime_strat():
        def regime_adaptive_strategy(df):
            signals = pd.Series(0, index=df.index)
            for regime in ["trending_up", "trending_down"]:
                mask = df["regime"] == regime
                if mask.sum() < 20:
                    continue
                regime_signals = donchian_breakout(df, 20)
                signals[mask] = regime_signals[mask]
            # Ranging: rsi_extreme
            mask_r = df["regime"] == "ranging"
            if mask_r.sum() >= 20:
                regime_signals = rsi_extreme(df)
                signals[mask_r] = regime_signals[mask_r]
            return signals
        return regime_adaptive_strategy

    for pair in ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"]:
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)

        bt = Backtester(pair, "regime_adaptive", make_regime_strat())
        portfolio, metrics = bt.run(df_full)

        trades = metrics.get("total_trades", 0)
        pnl = metrics.get("net_pnl", 0)
        wr = metrics.get("win_rate", 0)
        dd = metrics.get("max_drawdown", 0)
        pf = metrics.get("profit_factor", 0)
        final = metrics.get("final_balance", 0)
        print(f"  {pair}: {trades} trades | WR {wr:.0%} | PnL ${pnl:+.2f} | PF {pf:.2f} | DD {dd:.0%} | Final ${final:.2f}")

    # ============================================
    # TEST: Donchian-only (skip ranging entirely)
    # ============================================
    print(f"\n{'='*80}")
    print("DONCHIAN-ONLY (skip ranging)")
    print(f"{'='*80}\n")

    for pair in ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"]:
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)

        bt = Backtester(pair, "donchian_only", lambda df: donchian_breakout(df, 20))
        portfolio, metrics = bt.run(df_full)

        trades = metrics.get("total_trades", 0)
        pnl = metrics.get("net_pnl", 0)
        wr = metrics.get("win_rate", 0)
        dd = metrics.get("max_drawdown", 0)
        pf = metrics.get("profit_factor", 0)
        final = metrics.get("final_balance", 0)
        print(f"  {pair}: {trades} trades | WR {wr:.0%} | PnL ${pnl:+.2f} | PF {pf:.2f} | DD {dd:.0%} | Final ${final:.2f}")

    # ============================================
    # TEST: Trend-only Donchian (only trade trending regimes)
    # ============================================
    print(f"\n{'='*80}")
    print("TREND-ONLY DONCHIAN (only trending regimes)")
    print(f"{'='*80}\n")

    for pair in ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"]:
        df_full = fetch_ohlcv(exchange, pair, "1h", 12)
        df_full = detect_regime(df_full)

        def trend_only_donchian(df):
            signals = pd.Series(0, index=df.index)
            for regime in ["trending_up", "trending_down"]:
                mask = df["regime"] == regime
                if mask.sum() < 20:
                    continue
                regime_signals = donchian_breakout(df, 20)
                signals[mask] = regime_signals[mask]
            return signals

        bt = Backtester(pair, "trend_only_donchian", trend_only_donchian)
        portfolio, metrics = bt.run(df_full)

        trades = metrics.get("total_trades", 0)
        pnl = metrics.get("net_pnl", 0)
        wr = metrics.get("win_rate", 0)
        dd = metrics.get("max_drawdown", 0)
        pf = metrics.get("profit_factor", 0)
        final = metrics.get("final_balance", 0)
        print(f"  {pair}: {trades} trades | WR {wr:.0%} | PnL ${pnl:+.2f} | PF {pf:.2f} | DD {dd:.0%} | Final ${final:.2f}")

    # Save
    df_valid.to_csv("/tmp/regime_results.csv", index=False)
    print(f"\nFull results saved to /tmp/regime_results.csv")


if __name__ == "__main__":
    run_regime_tests()
