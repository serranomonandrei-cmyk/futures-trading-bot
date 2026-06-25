"""
Final bot: Donchian breakout ONLY during trending regimes.
Cash during ranging periods.

This is the honest answer after testing 30+ ranging strategies.
Ranging markets have no edge. Trends have edge. Trade trends, skip ranges.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
from strategies import donchian_breakout


def trend_only_bot(df):
    """
    Trade Donchian breakout only in trending regimes.
    Cash in ranging/volatile regimes.
    """
    signals = pd.Series(0, index=df.index)
    for regime in ["trending_up", "trending_down"]:
        mask = df["regime"] == regime
        if mask.sum() < 10:
            continue
        donchian_signals = donchian_breakout(df, 20)
        signals[mask] = donchian_signals[mask]
    return signals


def run_final():
    exchange = get_exchange()
    results = []

    print("=" * 70)
    print("  FINAL BOT: Trend-Only Donchian (ETH + BNB)")
    print("  Cash during ranges. Donchian during trends.")
    print("=" * 70)

    for pair in ["ETH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT"]:
        df = fetch_ohlcv(exchange, pair, "1h", 12)
        df = detect_regime(df)

        bt = Backtester(pair, "trend_only", trend_only_bot)
        portfolio, metrics = bt.run(df)

        t = metrics.get("total_trades", 0)
        wr = metrics.get("win_rate", 0)
        pnl = metrics.get("net_pnl", 0)
        dd = metrics.get("max_drawdown", 0)
        pf = metrics.get("profit_factor", 0)
        final = metrics.get("final_balance", 0)
        ret = metrics.get("return_pct", 0)

        print(f"\n  {pair}")
        print(f"    Trades: {t} | WR: {wr:.0%} | PnL: ${pnl:+.2f}")
        print(f"    PF: {pf:.2f} | DD: {dd:.0%} | Return: {ret:+.1f}%")
        print(f"    Final: ${final:.2f}")

        results.append({
            "pair": pair, "trades": t, "win_rate": wr,
            "pnl": pnl, "pf": pf, "max_dd": dd,
            "return_pct": ret, "final": final
        })

    # Compare to Donchian-only (no regime filter)
    print(f"\n{'='*70}")
    print("  COMPARISON: Donchian-only (no regime filter)")
    print(f"{'='*70}")

    for pair in ["ETH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT"]:
        df = fetch_ohlcv(exchange, pair, "1h", 12)
        bt = Backtester(pair, "donchian_no_regime", lambda df: donchian_breakout(df, 20))
        _, metrics = bt.run(df)
        t = metrics.get("total_trades", 0)
        pnl = metrics.get("net_pnl", 0)
        dd = metrics.get("max_drawdown", 0)
        final = metrics.get("final_balance", 0)
        ret = metrics.get("return_pct", 0)
        print(f"  {pair}: {t} trades | PnL ${pnl:+.2f} | DD {dd:.0%} | {ret:+.1f}% | Final ${final:.2f}")

    # Risk metrics
    print(f"\n{'='*70}")
    print("  RISK ANALYSIS")
    print(f"{'='*70}")

    for pair in ["ETH/USDT:USDT", "BNB/USDT:USDT"]:
        df = fetch_ohlcv(exchange, pair, "1h", 12)
        df = detect_regime(df)
        bt = Backtester(pair, "trend_only", trend_only_bot)
        portfolio, metrics = bt.run(df)

        # Count regimes
        regime_counts = df["regime"].value_counts()
        trending_pct = (regime_counts.get("trending_up", 0) + regime_counts.get("trending_down", 0)) / len(df) * 100
        ranging_pct = regime_counts.get("ranging", 0) / len(df) * 100

        print(f"\n  {pair}:")
        print(f"    Trending: {trending_pct:.0f}% | Ranging: {ranging_pct:.0f}%")
        print(f"    Trading only {trending_pct:.0f}% of the time")
        print(f"    Expected: ~{t * (100/trending_pct):.0f} trades if traded 100% of time")

    # Save
    pd.DataFrame(results).to_csv("/tmp/final_bot_results.csv", index=False)
    print(f"\nResults saved to /tmp/final_bot_results.csv")


if __name__ == "__main__":
    run_final()
