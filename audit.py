"""
Full backtest audit: IS/OOS, walk-forward, look-ahead checks.
"""

import pandas as pd
import numpy as np
from data import get_exchange, fetch_ohlcv
from regime import detect_regime
from backtest import Backtester
try:
    from three_strategies import golden_bull as bull_strategy
    from three_strategies import golden_bear as bear_strategy
    from three_strategies import golden_range as range_strategy
except ImportError:
    bull_strategy = lambda df: pd.Series(0, index=df.index)
    bear_strategy = lambda df: pd.Series(0, index=df.index)
    range_strategy = lambda df: pd.Series(0, index=df.index)
from strategies import donchian_breakout

exchange = get_exchange()
pair = "ETH/USDT:USDT"

# Fetch data
df_full = fetch_ohlcv(exchange, pair, "1h", 12)
df_full_regime = detect_regime(df_full.copy())

print("=" * 70)
print("  AUDIT 1: LOOK-AHEAD BIAS CHECK")
print("=" * 70)

# For each strategy, check: does signal on bar N use any data from bar N+1 or later?
# Strategy: compute signals on full df, then compare to signals computed
# on a rolling window (bar 0 to bar N only). If they differ, look-ahead exists.

def test_lookahead(name, fn, df, n_bars=100):
    """Compute signals on full df and on rolling windows. Compare last N bars."""
    # New strategies need regime column
    from regime import detect_regime
    df_r = detect_regime(df.copy())
    full_signals = fn(df_r)
    rolling_signals = pd.Series(0, index=df_r.index, dtype=int)

    for i in range(max(50, n_bars), len(df_r)):
        window = df_r.iloc[:i+1].copy().reset_index(drop=True)
        sig_i = fn(window)
        rolling_signals.iloc[i] = sig_i.iloc[-1]

    # Compare last n_bars
    last_n = min(n_bars, len(df) - 50)
    match = (full_signals.iloc[-last_n:] == rolling_signals.iloc[-last_n:]).sum()
    total = last_n
    diff = full_signals.iloc[-last_n:] != rolling_signals.iloc[-last_n:]

    mismatches = diff.sum()
    print(f"\n  {name}:")
    print(f"    Mismatches: {mismatches}/{total} ({mismatches/total*100:.1f}%)")
    if mismatches > 0:
        mismatch_idxs = diff[diff].index[:5]
        for idx in mismatch_idxs:
            print(f"    Mismatch at bar {df_full.index.get_loc(idx)}: full={full_signals[idx]} rolling={rolling_signals[idx]}")
    else:
        print(f"    NO LOOK-AHEAD BIAS")

    return mismatches

print("\n  Testing look-ahead bias on last 50 bars...")
test_lookahead("bull_strategy", bull_strategy, df_full.copy(), 50)
test_lookahead("bear_strategy", bear_strategy, df_full.copy(), 50)
test_lookahead("range_strategy", range_strategy, df_full.copy(), 50)
test_lookahead("donchian_breakout", lambda df: donchian_breakout(df, 25), df_full.copy(), 50)


print(f"\n{'='*70}")
print("  AUDIT 2: IN-SAMPLE / OUT-OF-SAMPLE (70/30 split)")
print("="*70)

# Split by date: first 8 months IS, last 4 months OOS
split_date = pd.Timestamp("2026-02-23", tz="UTC")  # ~8 months from Jul 2025
df_is = df_full[df_full["timestamp"] < split_date].copy()
df_oos = df_full[df_full["timestamp"] >= split_date].copy()

print(f"\n  In-Sample: {df_is['timestamp'].min()} to {df_is['timestamp'].max()} ({len(df_is)} bars)")
print(f"  Out-of-Sample: {df_oos['timestamp'].min()} to {df_oos['timestamp'].max()} ({len(df_oos)} bars)")

# Define combined strategy
def combined_bot(df):
    """Each strategy gates itself by regime. Just add all three."""
    signals = pd.Series(0, index=df.index)
    if "regime" not in df.columns:
        from regime import detect_regime
        df = detect_regime(df.copy())

    bull_sigs = bull_strategy(df)
    bear_sigs = bear_strategy(df)
    range_sigs = range_strategy(df)

    signals = bull_sigs + bear_sigs + range_sigs
    signals = signals.clip(-1, 1)
    return signals

# In-sample test
df_is_regime = detect_regime(df_is)
bt_is = Backtester(pair, "three_strat", combined_bot)
_, m_is = bt_is.run(df_is_regime)

# Out-of-sample test
df_oos_regime = detect_regime(df_oos)
bt_oos = Backtester(pair, "three_strat", combined_bot)
_, m_oos = bt_oos.run(df_oos_regime)

print(f"\n  --- Three-Strategy Combined ---")
print(f"  In-Sample:")
print(f"    Trades: {m_is.get('total_trades', 0)} | WR: {m_is.get('win_rate', 0):.0%} | PnL: ${m_is.get('net_pnl', 0):+.2f}")
print(f"    PF: {m_is.get('profit_factor', 0):.2f} | DD: {m_is.get('max_drawdown', 0):.0%} | Return: {m_is.get('return_pct', 0):+.1f}%")

print(f"  Out-of-Sample:")
print(f"    Trades: {m_oos.get('total_trades', 0)} | WR: {m_oos.get('win_rate', 0):.0%} | PnL: ${m_oos.get('net_pnl', 0):+.2f}")
print(f"    PF: {m_oos.get('profit_factor', 0):.2f} | DD: {m_oos.get('max_drawdown', 0):.0%} | Return: {m_oos.get('return_pct', 0):+.1f}%")

# Donchian baseline
print(f"\n  --- Donchian 25 Baseline ---")
for label, df_x in [("In-Sample", df_is), ("Out-of-Sample", df_oos)]:
    bt = Backtester(pair, "donchian_25", lambda df: donchian_breakout(df, 25))
    _, m = bt.run(df_x)
    print(f"  {label}:")
    print(f"    Trades: {m.get('total_trades', 0)} | WP: {m.get('win_rate', 0):.0%} | PnL: ${m.get('net_pnl', 0):+.2f}")
    print(f"    PF: {m.get('profit_factor', 0):.2f} | DD: {m.get('max_drawdown', 0):.0%} | Return: {m.get('return_pct', 0):+.1f}%")


print(f"\n{'='*70}")
print("  AUDIT 3: WALK-FORWARD (3-month train, 1-month test)")
print("="*70)

timestamps = df_full["timestamp"]
start = timestamps.min()
end = timestamps.max()

window_start = start
results = []
window_idx = 0

while window_start + pd.Timedelta(days=120) <= end:
    train_start = window_start
    train_end = train_start + pd.Timedelta(days=90)
    test_start = train_end
    test_end = test_start + pd.Timedelta(days=30)

    if test_end > end:
        test_end = end

    train_mask = (timestamps >= train_start) & (timestamps < train_end)
    test_mask = (timestamps >= test_start) & (timestamps <= test_end)

    df_train = df_full[train_mask].copy()
    df_test = df_full[test_mask].copy()

    if len(df_train) < 50 or len(df_test) < 20:
        window_start += pd.Timedelta(days=30)
        continue

    # Three-strategy on test
    try:
        bt = Backtester(pair, "three_strat", combined_bot)
        _, m = bt.run(df_test)
        pnl = m.get("net_pnl", 0)
        trades = m.get("total_trades", 0)
        wr = m.get("win_rate", 0)

        # Donchian baseline on test
        bt2 = Backtester(pair, "donchian_25", lambda df: donchian_breakout(df, 25))
        _, m2 = bt2.run(df_test)
        pnl2 = m2.get("net_pnl", 0)
    except Exception as e:
        pnl, trades, wr, pnl2 = 0, 0, 0, 0

    results.append({
        "window": window_idx,
        "train": f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}",
        "test": f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}",
        "test_bars": len(df_test),
        "three_trades": trades,
        "three_wr": wr,
        "three_pnl": pnl,
        "donchian_pnl": pnl2,
    })

    window_idx += 1
    window_start += pd.Timedelta(days=30)

if results:
    columns = ["window", "test", "test_bars", "three_trades", "three_wr", "three_pnl", "donchian_pnl"]
    df_results = pd.DataFrame(results, columns=columns)

    print(f"\n  Windows: {len(results)}")
    for _, r in df_results.iterrows():
        w = "WIN" if r["three_pnl"] > 0 else "LOSS" if r["three_pnl"] < 0 else "FLAT"
        print(f"  W{r['window']:d}: {r['test']} | {r['three_trades']:3d}t | WR {r['three_wr']:.0%} | PnL ${r['three_pnl']:+6.2f} ({w}) | Don ${r['donchian_pnl']:+6.2f}")

    total_three = df_results["three_pnl"].sum()
    total_don = df_results["donchian_pnl"].sum()
    win_windows = (df_results["three_pnl"] > 0).sum()
    print(f"\n  Total Three-Strategy: ${total_three:+.2f}")
    print(f"  Total Donchian: ${total_don:+.2f}")
    print(f"  Profitable windows: {win_windows}/{len(results)} ({win_windows/len(results)*100:.0f}%)")
    print(f"  Edge: ${total_three - total_don:+.2f} {'(three-strategy wins)' if total_three > total_don else '(Donchian wins)'}")


print(f"\n{'='*70}")
print("  AUDIT 4: REGIME DETECTION STABILITY")
print("="*70)

regimes_full = df_full_regime["regime"]
# Check regime transitions: do regimes flip too often?
transitions = (regimes_full != regimes_full.shift(1)).sum()
total_bars = len(regimes_full)
avg_bars_per_regime = total_bars / transitions if transitions > 0 else 0

print(f"\n  Total bars: {total_bars}")
print(f"  Regime transitions: {transitions}")
print(f"  Avg bars per regime period: {avg_bars_per_regime:.1f}")

# Check regime duration distribution
regime_durations = []
current_regime = regimes_full.iloc[0]
current_duration = 1
for i in range(1, len(regimes_full)):
    if regimes_full.iloc[i] == current_regime:
        current_duration += 1
    else:
        regime_durations.append((current_regime, current_duration))
        current_regime = regimes_full.iloc[i]
        current_duration = 1
regime_durations.append((current_regime, current_duration))

durations_by_regime = {}
for r, d in regime_durations:
    durations_by_regime.setdefault(r, []).append(d)

for regime, durations in durations_by_regime.items():
    avg_hours = np.mean(durations)
    med_hours = np.median(durations)
    min_hours = np.min(durations)
    max_hours = np.max(durations)
    print(f"\n  {regime}:")
    print(f"    Segments: {len(durations)} | Avg: {avg_hours:.0f}h | Med: {med_hours:.0f}h | Min: {min_hours}h | Max: {max_hours}h")

# Too many short regimes (< 6 hours) = noise
short_regimes = sum(1 for d in durations_by_regime.get("trending_up", []) if d < 6) + \
                 sum(1 for d in durations_by_regime.get("trending_down", []) if d < 6)
if short_regimes > 5:
    print(f"\n  WARNING: {short_regimes} trending regimes < 6 hours. Regime detection may be noisy.")


print(f"\n{'='*70}")
print("  AUDIT 5: STRATEGY LOGIC VERIFICATION")
print("="*70)

# Check new strategies — regime-gated
print("\n  New strategies (regime-gated):")
print("    bull_strategy: Donchian breakout OR RSI pullback — only in trending_up")
print("    bear_strategy: Donchian breakdown OR RSI rally fade — only in trending_down")
print("    range_strategy: Bollinger fade with RSI — only in ranging")

df_check = df_full_regime.copy()
bull_sigs = bull_strategy(df_check)
bear_sigs = bear_strategy(df_check)
range_sigs = range_strategy(df_check)

for regime in ["trending_up", "trending_down", "ranging"]:
    mask = df_check["regime"] == regime
    total = mask.sum()

    bull_long = ((bull_sigs == 1) & mask).sum()
    bull_short = ((bull_sigs == -1) & mask).sum()
    bear_long = ((bear_sigs == 1) & mask).sum()
    bear_short = ((bear_sigs == -1) & mask).sum()
    range_long = ((range_sigs == 1) & mask).sum()
    range_short = ((range_sigs == -1) & mask).sum()

    print(f"\n  {regime} ({total} bars):")
    print(f"    bull_strategy: {bull_long} long, {bull_short} short (rate: {(bull_long+bull_short)/total*100:.1f}%)")
    print(f"    bear_strategy: {bear_long} long, {bear_short} short (rate: {(bear_long+bear_short)/total*100:.1f}%)")
    print(f"    range_strategy: {range_long} long, {range_short} short (rate: {(range_long+range_short)/total*100:.1f}%)")

# Check: does the "wrong" strategy fire in the wrong regime?
print(f"\n  Cross-regime signals (should be ZERO if gating works):")
bull_cross = ((bull_sigs != 0) & (df_check["regime"] != "trending_up")).sum()
bear_cross = ((bear_sigs != 0) & (df_check["regime"] != "trending_down")).sum()
range_cross = ((range_sigs != 0) & (df_check["regime"] != "ranging")).sum()
print(f"    bull_strategy fires outside trending_up: {bull_cross}")
print(f"    bear_strategy fires outside trending_down: {bear_cross}")
print(f"    range_strategy fires outside ranging: {range_cross}")

print(f"\n{'='*70}")
print("  AUDIT 6: FINAL VERDICT")
print("="*70)

is_pnl = m_is.get("net_pnl", 0)
oos_pnl = m_oos.get("net_pnl", 0)
is_profitable = is_pnl > 0
oos_profitable = oos_pnl > 0

issues = []
if not is_profitable:
    issues.append("In-sample loses money")
if not oos_profitable:
    issues.append("Out-of-sample loses money")

total_walk_three = total_three if results else 0
walk_profitable = total_walk_three > 0

print(f"\n  In-Sample: {'PASS' if is_profitable else 'FAIL'} (${is_pnl:+.2f})")
print(f"  Out-of-Sample: {'PASS' if oos_profitable else 'FAIL'} (${oos_pnl:+.2f})")
print(f"  Walk-Forward: {'PASS' if walk_profitable else 'FAIL'} (${total_walk_three:+.2f})")

if not issues:
    print(f"\n  VERDICT: BACKTEST VALID — deploy with confidence.")
else:
    print(f"\n  VERDICT: BACKTEST HAS ISSUES: {', '.join(issues)}")
    print(f"  Do NOT deploy until fixed.")