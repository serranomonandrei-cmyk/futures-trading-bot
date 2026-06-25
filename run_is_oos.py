"""
In-sample / Out-of-sample + Per-Regime breakdown.
70/30 split. No data leakage.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv, resample_ohlcv
from regime import detect_regime
from strategies import ALL_STRATEGIES
from backtest import Backtester, Portfolio, calc_stop, calc_tp, calc_position_size, apply_slippage, Trade
from config import PAIRS, STARTING_BALANCE, TAKER_FEE_PCT, FUNDING_RATE_PCT, MAX_DRAWDOWN_PCT, MAX_CONCURRENT_POSITIONS, MIN_STOP_DISTANCE_PCT


class BacktesterWithRegime(Backtester):
    """Extended backtester that tracks per-regime performance."""

    def run(self, df):
        portfolio = Portfolio()
        signals = self.strategy_fn(df)

        from regime import calc_atr
        df = df.copy()
        df["atr"] = calc_atr(df, 14)

        # Track per-regime stats
        regime_stats = {}  # regime -> {"trades": [], "pnl": 0, "wins": 0, "losses": 0}

        for i in range(1, len(df)):
            signal = signals.iloc[i-1]
            self._check_positions(portfolio, df, i)

            if signal != 0 and len(portfolio.open_positions) < MAX_CONCURRENT_POSITIONS:
                self._execute_entry(portfolio, df, i, signal)

            portfolio.peak_balance = max(portfolio.peak_balance, portfolio.balance)

        self._close_all(portfolio, df, len(df)-1)

        # Build regime breakdown from trade entry times
        for trade in portfolio.trades:
            # Find regime at entry time
            if hasattr(trade.entry_time, 'timestamp'):
                mask = df["timestamp"] == trade.entry_time
                if mask.any():
                    regime = df.loc[mask, "regime"].iloc[0]
                else:
                    regime = "unknown"
            else:
                regime = "unknown"

            if regime not in regime_stats:
                regime_stats[regime] = {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}

            regime_stats[regime]["trades"] += 1
            regime_stats[regime]["pnl"] += trade.pnl
            if trade.pnl > 0:
                regime_stats[regime]["wins"] += 1
            else:
                regime_stats[regime]["losses"] += 1

        metrics = self._calc_metrics(portfolio, df)
        metrics["regime_breakdown"] = regime_stats
        return portfolio, metrics


def run_split_analysis():
    exchange = get_exchange()
    IS_RATIO = 0.70  # 70% in-sample, 30% out-of-sample

    # Top 10 strategies from previous run
    top_strategies = [
        "ema_rsi_volume", "hull_ma", "stochastic_extreme", "volume_breakout",
        "donchian_20", "triple_ema", "ema_cross_5_13", "inside_bar",
        "parabolic_sar", "adx_di"
    ]

    all_results = []

    for pair in PAIRS:
        print(f"\n{'='*80}")
        print(f"  {pair}")
        print(f"{'='*80}")

        # Fetch 12 months for more data (better IS/OOS split)
        df_1h = fetch_ohlcv(exchange, pair, "1h", 12)
        df_1h = detect_regime(df_1h)

        # Split: 70% IS, 30% OOS
        split_idx = int(len(df_1h) * IS_RATIO)
        df_is = df_1h.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_1h.iloc[split_idx:].copy().reset_index(drop=True)

        print(f"  Total bars: {len(df_1h)}")
        print(f"  In-sample:  {len(df_is)} bars ({df_is['timestamp'].iloc[0].date()} to {df_is['timestamp'].iloc[-1].date()})")
        print(f"  Out-of-sample: {len(df_oos)} bars ({df_oos['timestamp'].iloc[0].date()} to {df_oos['timestamp'].iloc[-1].date()})")

        # Regime distribution in each split
        for name, df_split, split_label in [("IS", df_is, "in-sample"), ("OOS", df_oos, "out-of-sample")]:
            regimes = df_split[df_split["regime"] != "unknown"]["regime"].value_counts()
            total = regimes.sum()
            parts = []
            for r in ["trending_up", "trending_down", "ranging", "volatile"]:
                if r in regimes.index:
                    parts.append(f"{r}:{regimes[r]/total*100:.0f}%")
            print(f"    {split_label}: {', '.join(parts)}")

        for strat_name in top_strategies:
            strategy_fn = ALL_STRATEGIES[strat_name]

            # Run on in-sample
            bt_is = BacktesterWithRegime(pair, strat_name, strategy_fn)
            _, metrics_is = bt_is.run(df_is)

            # Run on out-of-sample
            bt_oos = BacktesterWithRegime(pair, strat_name, strategy_fn)
            _, metrics_oos = bt_oos.run(df_oos)

            result = {
                "pair": pair,
                "strategy": strat_name,
                # In-sample
                "is_trades": metrics_is.get("total_trades", 0),
                "is_wr": metrics_is.get("win_rate", 0),
                "is_pnl": metrics_is.get("net_pnl", 0),
                "is_pf": metrics_is.get("profit_factor", 0),
                "is_dd": metrics_is.get("max_drawdown", 0),
                "is_sharpe": metrics_is.get("sharpe", 0),
                "is_final": metrics_is.get("final_balance", 0),
                "is_regimes": metrics_is.get("regime_breakdown", {}),
                # Out-of-sample
                "oos_trades": metrics_oos.get("total_trades", 0),
                "oos_wr": metrics_oos.get("win_rate", 0),
                "oos_pnl": metrics_oos.get("net_pnl", 0),
                "oos_pf": metrics_oos.get("profit_factor", 0),
                "oos_dd": metrics_oos.get("max_drawdown", 0),
                "oos_sharpe": metrics_oos.get("sharpe", 0),
                "oos_final": metrics_oos.get("final_balance", 0),
                "oos_regimes": metrics_oos.get("regime_breakdown", {}),
            }
            all_results.append(result)

            # Quick line
            is_pnl = metrics_is.get("net_pnl", 0)
            oos_pnl = metrics_oos.get("net_pnl", 0)
            degradation = ((oos_pnl - is_pnl) / abs(is_pnl) * 100) if is_pnl != 0 else 0
            flag = "OK" if degradation > -50 else "WARN" if degradation > -80 else "OVERFIT"
            print(f"    {strat_name:25s} | IS: ${is_pnl:+6.2f} ({metrics_is.get('win_rate',0):.0%}) | OOS: ${oos_pnl:+6.2f} ({metrics_oos.get('win_rate',0):.0%}) | {flag}")

    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("IN-SAMPLE vs OUT-OF-SAMPLE SUMMARY")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Sort by OOS PnL
    df = df.sort_values("oos_pnl", ascending=False)

    print(f"{'Strategy':25s} {'Pair':15s} {'IS PnL':>8s} {'IS WR':>6s} {'OOS PnL':>8s} {'OOS WR':>7s} {'Degrad':>8s} {'Status':>8s}")
    print("-" * 95)

    for _, row in df.iterrows():
        deg = ((row["oos_pnl"] - row["is_pnl"]) / abs(row["is_pnl"]) * 100) if row["is_pnl"] != 0 else 0
        flag = "OK" if deg > -50 else "WARN" if deg > -80 else "OVERFIT"
        print(f"{row['strategy']:25s} {row['pair']:15s} ${row['is_pnl']:+7.2f} {row['is_wr']:5.0%} ${row['oos_pnl']:+7.2f} {row['oos_wr']:6.0%} {deg:+7.0f}% {flag:>8s}")

    # === PER-REGIME BREAKDOWN ===
    print(f"\n{'='*80}")
    print("PER-REGIME PERFORMANCE (Out-of-Sample)")
    print(f"{'='*80}\n")

    for _, row in df.head(10).iterrows():
        print(f"  {row['strategy']} on {row['pair']}:")
        oos_regimes = row["oos_regimes"]
        if oos_regimes:
            for regime in ["trending_up", "trending_down", "ranging", "volatile"]:
                if regime in oos_regimes:
                    rs = oos_regimes[regime]
                    wr = rs["wins"] / rs["trades"] * 100 if rs["trades"] > 0 else 0
                    avg_pnl = rs["pnl"] / rs["trades"] if rs["trades"] > 0 else 0
                    print(f"    {regime:15s}: {rs['trades']:3d} trades | WR {wr:.0f}% | PnL ${rs['pnl']:+.2f} | Avg ${avg_pnl:+.2f}")
        else:
            print(f"    (no regime data)")
        print()

    # === WALK-FORWARD EFFICIENCY ===
    print(f"{'='*80}")
    print("WALK-FORWARD EFFICIENCY (OOS/IS ratio)")
    print(f"{'='*80}\n")

    for _, row in df.head(10).iterrows():
        if row["is_pnl"] != 0:
            wfe = row["oos_pnl"] / row["is_pnl"] * 100
        else:
            wfe = 0
        quality = "EXCELLENT" if wfe > 70 else "GOOD" if wfe > 40 else "WEAK" if wfe > 0 else "OVERFIT"
        print(f"  {row['strategy']:25s} {row['pair']:15s} | WFE: {wfe:+.0f}% ({quality})")

    # Save
    df.to_csv("/tmp/is_oos_results.csv", index=False)
    print(f"\nFull results saved to /tmp/is_oos_results.csv")


if __name__ == "__main__":
    run_split_analysis()
