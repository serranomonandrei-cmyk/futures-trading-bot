"""
Test each strategy on its regime, then combine.
"""

import pandas as pd
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
from three_strategies import bull_pullback, bear_breakdown, range_reversion
from strategies import donchian_breakout

exchange = get_exchange()

# Hardcoded regime periods (from data analysis)
PERIODS = {
    "ETH/USDT:USDT": {
        "trending_up": [
            ("2025-07-14", "2025-07-18"),
            ("2025-08-06", "2025-08-14"),
            ("2025-12-01", "2025-12-05"),
        ],
        "trending_down": [
            ("2026-01-19", "2026-01-22"),
            ("2026-01-30", "2026-02-03"),
            ("2026-02-26", "2026-03-01"),
            ("2026-06-04", "2026-06-07"),
        ],
        "ranging": [
            ("2025-07-01", "2025-07-13"),
            ("2025-09-01", "2025-11-30"),
            ("2026-03-01", "2026-05-31"),
        ],
    },
}

print("=" * 70)
print("  TEST 1: Each strategy on its specific regime periods")
print("=" * 70)

pair = "ETH/USDT:USDT"
df_1h = fetch_ohlcv(exchange, pair, "1h", 12)

for regime, periods in PERIODS[pair].items():
    if regime == "trending_up":
        strats = {"bull_pullback": bull_pullback, "donchian_20": lambda df: donchian_breakout(df, 20)}
    elif regime == "trending_down":
        strats = {"bear_breakdown": bear_breakdown, "donchian_20": lambda df: donchian_breakout(df, 20)}
    else:
        strats = {"range_reversion": range_reversion}

    print(f"\n  --- {regime.upper()} ---")

    for strat_name, strat_fn in strats.items():
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        max_dd = 0

        for start_str, end_str in periods:
            start = pd.Timestamp(start_str, tz="UTC")
            end = pd.Timestamp(end_str, tz="UTC")
            mask = (df_1h["timestamp"] >= start) & (df_1h["timestamp"] <= end)
            df_p = df_1h[mask].copy().reset_index(drop=True)
            if len(df_p) < 10:
                continue

            bt = Backtester(pair, strat_name, strat_fn)
            try:
                _, m = bt.run(df_p)
                total_pnl += m.get("net_pnl", 0)
                total_trades += m.get("total_trades", 0)
                total_wins += m.get("wins", 0)
                max_dd = max(max_dd, m.get("max_drawdown", 0))
            except:
                pass

        wr = total_wins / total_trades if total_trades > 0 else 0
        flag = "OK" if total_pnl > 0 else "LOSS" if total_trades > 0 else "NO TRADES"
        print(f"    {strat_name:20s} | {total_trades:3d} trades | WR {wr:.0%} | PnL ${total_pnl:+6.2f} | DD {max_dd:.0%} | {flag}")


# ============================================================
# TEST 2: Combined bot — bull + bear + ranging, full 12 months
# ============================================================
print(f"\n{'='*70}")
print("  TEST 2: Combined three-strategy bot (full 12 months)")
print("="*70)

def three_strategy_bot(df):
    """Donchian for all trending, range reversion for ranging.
    Donchian is NOT regime-filtered — it naturally avoids ranges."""
    signals = pd.Series(0, index=df.index)

    if "regime" not in df.columns:
        from regime import detect_regime
        df = detect_regime(df.copy())

    uptrend = df["regime"] == "trending_up"
    downtrend = df["regime"] == "trending_down"
    ranging = df["regime"] == "ranging"

    # Donchian for ALL data (no filter — it naturally avoids ranges)
    donchian_sigs = donchian_breakout(df, 25)
    signals = donchian_sigs.copy()

    # Override ranging with range reversion
    range_sigs = range_reversion(df)
    signals[ranging] = range_sigs[ranging]

    return signals


# Need ema import
from three_strategies import ema

bt = Backtester(pair, "three_strategy", three_strategy_bot)
_, m = bt.run(df_1h)
t = m.get("total_trades", 0)
wr = m.get("win_rate", 0)
pnl = m.get("net_pnl", 0)
dd = m.get("max_drawdown", 0)
pf = m.get("profit_factor", 0)
final = m.get("final_balance", 0)
ret = m.get("return_pct", 0)

print(f"\n  Three-Strategy Combined:")
print(f"    Trades: {t} | WR: {wr:.0%} | PnL: ${pnl:+.2f}")
print(f"    PF: {pf:.2f} | DD: {dd:.0%} | Return: {ret:+.1f}%")
print(f"    Final: ${final:.2f}")


# ============================================================
# TEST 3: Compare to Donchian-only baseline
# ============================================================
print(f"\n{'='*70}")
print("  TEST 3: Baseline comparison (Donchian only)")
print("="*70)

bt2 = Backtester(pair, "donchian_25", lambda df: donchian_breakout(df, 25))
_, m2 = bt2.run(df_1h)
t2 = m2.get("total_trades", 0)
pnl2 = m2.get("net_pnl", 0)
dd2 = m2.get("max_drawdown", 0)
final2 = m2.get("final_balance", 0)
ret2 = m2.get("return_pct", 0)

print(f"\n  Donchian 25 (baseline):")
print(f"    Trades: {t2} | PnL: ${pnl2:+.2f} | DD: {dd2:.0%} | Return: {ret2:+.1f}% | Final: ${final2:.2f}")

print(f"\n  Three-Strategy:")
print(f"    Trades: {t} | PnL: ${pnl:+.2f} | DD: {dd:.0%} | Return: {ret:+.1f}% | Final: ${final:.2f}")

edge = pnl - pnl2
print(f"\n  Edge: ${edge:+.2f} {'(three-strategy wins)' if edge > 0 else '(Donchian wins)'}")


# ============================================================
# TEST 4: Per-regime performance breakdown
# ============================================================
print(f"\n{'='*70}")
print("  TEST 4: Per-regime performance (three-strategy bot)")
print("="*70)

from regime import detect_regime
df_1h_regime = detect_regime(df_1h.copy())

for regime in ["trending_up", "trending_down", "ranging"]:
    mask = df_1h_regime["regime"] == regime
    df_r = df_1h_regime[mask].copy().reset_index(drop=True)
    if len(df_r) < 50:
        print(f"\n  {regime}: insufficient data ({len(df_r)} bars)")
        continue

    bt_r = Backtester(pair, "three_strategy", three_strategy_bot)
    try:
        _, m_r = bt_r.run(df_r)
        t_r = m_r.get("total_trades", 0)
        wr_r = m_r.get("win_rate", 0)
        pnl_r = m_r.get("net_pnl", 0)
        dd_r = m_r.get("max_drawdown", 0)
        pf_r = m_r.get("profit_factor", 0)
        print(f"\n  {regime}:")
        print(f"    Trades: {t_r} | WR: {wr_r:.0%} | PnL: ${pnl_r:+.2f} | PF: {pf_r:.2f} | DD: {dd_r:.0%}")
    except Exception as e:
        print(f"\n  {regime}: error - {e}")
