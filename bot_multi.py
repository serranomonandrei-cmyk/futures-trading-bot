"""
Multi-Coin Futures Bot — SMA(30) Crossover on 8 coins with one $20 account.
Strategy: Long when price crosses above SMA(30), Short when crosses below.
Max 3 concurrent positions across all coins. 72h time cap on positions.

Usage:
  python3 bot_multi.py           # Paper mode (default)
  python3 bot_multi.py --live    # Live trading
  python3 bot_multi.py --once    # Single tick check
"""

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (PAIRS, TF_CANDLES, TAKER_FEE_PCT, SLIPPAGE_PCT,
                    MAX_DRAWDOWN_PCT, MAX_CONCURRENT_POSITIONS, MAX_MARGIN_UTILIZATION,
                    MIN_STOP_DISTANCE_PCT, MAX_DAILY_TRADES)
from data import get_exchange, fetch_ohlcv

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Strategy params
SMA_PERIOD = 30
LEVERAGE = 15
RISK_PCT = 0.03
RR_RATIO = 4.0
ATR_STOP_MULT = 2.0
MAX_BARS_HELD = 72  # 3-day time cap


def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_signal(df):
    s = df["close"].rolling(SMA_PERIOD).mean()
    cross_up = (df["close"] > s) & (df["close"].shift(1) <= s.shift(1))
    cross_down = (df["close"] < s) & (df["close"].shift(1) >= s.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up] = 1
    sig[cross_down] = -1
    return sig.iloc[-1]


class Bot:
    def __init__(self, live=False):
        self.live = live
        self.exchange = get_exchange()
        self.pairs = PAIRS
        self.state_file = LOG_DIR / "bot_state.json"
        self.log_file = LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        self.candle_cache = {}
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            # Migrate from old single-coin bot format
            if "position" in self.state:
                del self.state["position"]
            if "positions" not in self.state:
                self.state["positions"] = {}
            for key in ["daily_trades", "last_day"]:
                if key not in self.state:
                    self.state[key] = 0 if key == "daily_trades" else ""
        else:
            self.state = {
                "balance": 20.0,
                "peak_balance": 20.0,
                "positions": {},
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "daily_trades": 0,
                "last_day": "",
                "start_time": datetime.now(timezone.utc).isoformat(),
            }
        self.save_state()

    def save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def log_trade(self, trade):
        trades = []
        if self.log_file.exists():
            trades = json.loads(self.log_file.read_text())
        trades.append(trade)
        self.log_file.write_text(json.dumps(trades, indent=2))

    def get_candles(self, pair):
        if pair not in self.candle_cache:
            self.candle_cache.clear()
        df = fetch_ohlcv(self.exchange, pair, TF_CANDLES, months=2)
        self.candle_cache[pair] = df
        return df

    def _check_position(self, pair, pos, df):
        now = datetime.now(timezone.utc)
        bar = df.iloc[-1]
        side = pos["side"]
        entry = pos["entry_price"]
        stop = pos["stop"]
        tp = pos["tp"]
        margin = pos["margin"]
        bars_held = pos.get("bars", 0) + 1
        pos["bars"] = bars_held

        # Time cap — force close after MAX_BARS_HELD
        hit_time = bars_held >= MAX_BARS_HELD
        hit_sl = False
        hit_tp = False
        exit_price = None
        reason = ""

        if hit_time:
            exit_price = bar["close"]
            reason = "TIME"
        elif side == "long":
            if bar["low"] <= stop:
                hit_sl = True
                exit_price = stop
                reason = "SL"
            elif bar["high"] >= tp:
                hit_tp = True
                exit_price = tp
                reason = "TP"
        else:
            if bar["high"] >= stop:
                hit_sl = True
                exit_price = stop
                reason = "SL"
            elif bar["low"] <= tp:
                hit_tp = True
                exit_price = tp
                reason = "TP"

        if exit_price is None:
            return False

        # Calculate PnL
        if side == "long":
            pnl_pct = (exit_price - entry) / entry - SLIPPAGE_PCT
        else:
            pnl_pct = (entry - exit_price) / entry - SLIPPAGE_PCT

        gross_pnl = pnl_pct * (margin * LEVERAGE)  # notional = margin * leverage
        fees = margin * LEVERAGE * TAKER_FEE_PCT * 2
        net_pnl = gross_pnl - fees

        self.state["balance"] += net_pnl
        self.state["total_pnl"] += net_pnl
        self.state["trades"] += 1
        if net_pnl > 0:
            self.state["wins"] += 1

        trade = {
            "time": now.isoformat(),
            "pair": pair.split("/")[0],
            "side": side,
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "pnl": round(net_pnl, 4),
            "reason": reason,
            "bars_held": bars_held,
            "balance": round(self.state["balance"], 2),
        }
        self.log_trade(trade)
        print(f"  CLOSED {pair.split('/')[0]} {reason}: {side} @ {exit_price:.4f} | PnL ${net_pnl:+.4f} | Bal ${self.state['balance']:.2f}")

        del self.state["positions"][pair]
        self.save_state()
        return True

    def _open_position(self, pair, df):
        now = datetime.now(timezone.utc)
        positions = self.state["positions"]
        bal = self.state["balance"]

        # Max concurrent positions
        if len(positions) >= MAX_CONCURRENT_POSITIONS:
            return

        if bal <= 0:
            return

        # Kill switch
        dd = (self.state["peak_balance"] - bal) / self.state["peak_balance"]
        if dd >= MAX_DRAWDOWN_PCT:
            return

        # Daily trade limit
        today = now.strftime("%Y-%m-%d")
        if self.state.get("last_day") == today and self.state.get("daily_trades", 0) >= MAX_DAILY_TRADES:
            return

        signal = calc_signal(df)
        if signal == 0:
            return

        side = "long" if signal == 1 else "short"
        entry = df.iloc[-1]["open"] * (1 + SLIPPAGE_PCT if side == "long" else 1 - SLIPPAGE_PCT)

        atr_val = calc_atr(df).iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return

        dist = max(atr_val * ATR_STOP_MULT, entry * MIN_STOP_DISTANCE_PCT)
        stop = entry - dist if side == "long" else entry + dist
        tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO

        risk_usd = bal * RISK_PCT
        stop_dist_pct = abs(entry - stop) / entry
        if stop_dist_pct < 1e-6:
            return

        notional = risk_usd / stop_dist_pct
        notional = min(notional, bal * 10)  # cap exposure
        margin = notional / LEVERAGE

        # Margin utilization check
        total_margin = sum(p["margin"] for p in positions.values())
        if total_margin + margin > bal * MAX_MARGIN_UTILIZATION:
            return

        # Minimum position size
        if margin < 1.0:
            return

        positions[pair] = {
            "side": side,
            "entry_price": entry,
            "stop": stop,
            "tp": tp,
            "leverage": LEVERAGE,
            "margin": margin,
            "bars": 0,
            "open_time": now.isoformat(),
        }
        self.state["peak_balance"] = max(self.state["peak_balance"], bal)

        # Daily trade counter
        if self.state.get("last_day") != today:
            self.state["daily_trades"] = 0
            self.state["last_day"] = today
        self.state["daily_trades"] += 1

        coin = pair.split("/")[0]
        trade = {
            "time": now.isoformat(),
            "action": "OPEN",
            "pair": coin,
            "side": side,
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "tp": round(tp, 2),
            "margin": round(margin, 2),
            "leverage": LEVERAGE,
            "balance": round(bal, 2),
        }
        self.log_trade(trade)
        print(f"  OPENED {coin} {side}: entry {entry:.2f} | stop {stop:.2f} | tp {tp:.2f} | margin ${margin:.2f}")
        self.save_state()

    def tick(self):
        now = datetime.now(timezone.utc)
        print(f"\n[{now.strftime('%H:%M:%S')}] Tick — {len(self.pairs)} pairs")

        if "positions" not in self.state:
            self.state["positions"] = {}

        # 1. Check/close existing positions
        for pair, pos in list(self.state["positions"].items()):
            try:
                df = self.get_candles(pair)
                self._check_position(pair, pos, df)
            except Exception as e:
                print(f"  CHECK {pair.split('/')[0]}: ERROR — {e}")

        # 2. Open new positions on available pairs
        for pair in self.pairs:
            if pair in self.state["positions"]:
                continue
            if len(self.state["positions"]) >= MAX_CONCURRENT_POSITIONS:
                break
            try:
                df = self.get_candles(pair)
                self._open_position(pair, df)
            except Exception as e:
                print(f"  OPEN {pair.split('/')[0]}: ERROR — {e}")

        # 3. Print status
        self._print_status()

    def _print_status(self):
        s = self.state
        dd = (s["peak_balance"] - s["balance"]) / s["peak_balance"] if s["peak_balance"] > 0 else 0
        wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        pos_count = len(s.get("positions", {}))

        print(f"\n  Balance: ${s['balance']:.2f} | PnL: ${s['total_pnl']:+.2f} | WR: {wr:.0%}/{s['trades']}t | DD: {dd:.0%}")
        print(f"  Positions: {pos_count}/{MAX_CONCURRENT_POSITIONS}")

        for pair, pos in s.get("positions", {}).items():
            coin = pair.split("/")[0]
            s_arrow = "↑" if pos["side"] == "long" else "↓"
            print(f"    {coin:6s} {s_arrow} {pos['side']:5s} @ {pos['entry_price']:.4f} | SL {pos['stop']:.4f} | TP {pos['tp']:.4f} | {pos.get('bars',0)}h")

    def run(self, interval_minutes=60):
        print(f"\n{'='*55}")
        print(f"  MULTI-COIN SMA({SMA_PERIOD}) BOT — {len(self.pairs)} pairs")
        print(f"  Account: $20 | Leverage: {LEVERAGE}x | Risk: {RISK_PCT:.0%}")
        print(f"  Max {MAX_CONCURRENT_POSITIONS} pos | {MAX_BARS_HELD}h cap | {MAX_MARGIN_UTILIZATION:.0%} margin")
        print(f"  Mode: {'LIVE' if self.live else 'PAPER'} | Bal: ${self.state['balance']:.2f}")
        print(f"  Pairs: {', '.join(p.split('/')[0] for p in self.pairs)}")
        print(f"{'='*55}\n")

        while True:
            self.tick()
            print(f"\n  Sleeping {interval_minutes}m...")
            time.sleep(interval_minutes * 60)

    def run_once(self):
        print(f"\n{'='*55}")
        print(f"  MULTI-COIN SMA({SMA_PERIOD}) — {len(self.pairs)} PAIRS")
        print(f"  Account: $20 | {LEVERAGE}x | {RISK_PCT:.0%} risk | {MAX_CONCURRENT_POSITIONS} max pos")
        print(f"{'='*55}\n")
        self.tick()


def main():
    live = "--live" in sys.argv
    once = "--once" in sys.argv
    bot = Bot(live=live)
    if once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()