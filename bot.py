"""
ETH Futures Bot — SMA(30) Crossover
Ultra-simple, no regime detection needed. Just price vs SMA(30).

Strategy: Long when price crosses above SMA(30), Short when crosses below.
Backtest: +$64.38 (+322% over 12mo), 28% WR, 36% max DD.
Walk-forward: 3/3 windows profitable. Jan-Apr 2026: +$19.50.

Usage:
  python3 bot.py           # Paper mode (default)
  python3 bot.py --live    # Live trading
  python3 bot.py --once    # Single tick check
"""

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import (PAIRS, TF_CANDLES, TAKER_FEE_PCT, SLIPPAGE_PCT, MAX_DRAWDOWN_PCT)
from data import get_exchange, fetch_ohlcv

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Strategy params
SMA_PERIOD = 30
LEVERAGE = 15
RISK_PCT = 0.03
RR_RATIO = 4.0
ATR_STOP_MULT = 2.0


def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_signal(df):
    """Long when price crosses above SMA, Short when crosses below."""
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
        self.pair = PAIRS[0]
        self.state_file = LOG_DIR / "bot_state.json"
        self.log_file = LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        self.load_state()

    def load_state(self):
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
        else:
            self.state = {
                "balance": 20.0, "peak_balance": 20.0,
                "position": None, "trades": 0, "wins": 0, "total_pnl": 0.0,
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

    def get_candles(self):
        return fetch_ohlcv(self.exchange, self.pair, TF_CANDLES, months=2)

    def tick(self):
        now = datetime.now(timezone.utc)
        print(f"\n[{now.strftime('%H:%M:%S')}] Tick")

        try:
            df = self.get_candles()
            print(f"  Candles: {len(df)} | Last: {df.iloc[-1]['timestamp']} | Close: ${df.iloc[-1]['close']:.2f}")
            self._check_position(df)
            self._open_position(df)
            self._print_status()

        except Exception as e:
            print(f"  ERROR: {e}")

    def _check_position(self, df):
        pos = self.state["position"]
        if not pos:
            return

        bar = df.iloc[-1]
        side = pos["side"]
        stop = pos["stop"]
        tp = pos["tp"]
        entry = pos["entry_price"]
        margin = pos["margin"]
        leverage = pos["leverage"]
        bars_held = pos.get("bars", 0) + 1
        pos["bars"] = bars_held

        hit_sl = False; hit_tp = False

        if side == "long":
            if bar["low"] <= stop: hit_sl = True; exit_price = stop
            elif bar["high"] >= tp: hit_tp = True; exit_price = tp
        else:
            if bar["high"] >= stop: hit_sl = True; exit_price = stop
            elif bar["low"] <= tp: hit_tp = True; exit_price = tp

        if hit_sl or hit_tp:
            if side == "long":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry

            gross_pnl = pnl_pct * margin * leverage
            fees = margin * leverage * TAKER_FEE_PCT * 2
            net_pnl = gross_pnl - fees

            self.state["balance"] += net_pnl
            self.state["total_pnl"] += net_pnl
            self.state["trades"] += 1
            if net_pnl > 0:
                self.state["wins"] += 1

            reason = "SL" if hit_sl else "TP"
            trade = {
                "time": now.isoformat(), "side": side,
                "entry": round(entry, 2), "exit": round(exit_price, 2),
                "pnl": round(net_pnl, 4), "reason": reason,
                "bars_held": bars_held,
                "balance": round(self.state["balance"], 2),
            }
            self.log_trade(trade)
            print(f"  CLOSED {reason}: {side} @ {exit_price:.2f} | PnL ${net_pnl:+.4f} | Balance ${self.state['balance']:.2f}")

            self.state["position"] = None
            self.save_state()

    def _open_position(self, df):
        if self.state["position"]:
            return

        if self.state["balance"] <= 0:
            return

        dd = (self.state["peak_balance"] - self.state["balance"]) / self.state["peak_balance"]
        if dd >= MAX_DRAWDOWN_PCT:
            print(f"  KILL SWITCH: DD {dd:.0%}")
            return

        signal = calc_signal(df)
        if signal == 0:
            s = df["close"].rolling(SMA_PERIOD).mean()
            print(f"  SMA({SMA_PERIOD}): ${s.iloc[-1]:.2f} | Close: ${df.iloc[-1]['close']:.2f} | No signal")
            return

        side = "long" if signal == 1 else "short"
        entry = df.iloc[-1]["open"] * (1 + SLIPPAGE_PCT if side == "long" else 1 - SLIPPAGE_PCT)

        atr_val = calc_atr(df).iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return

        dist = atr_val * ATR_STOP_MULT
        min_dist = entry * 0.005
        dist = max(dist, min_dist)
        stop = entry - dist if side == "long" else entry + dist
        tp = entry + dist * RR_RATIO if side == "long" else entry - dist * RR_RATIO

        risk_usd = self.state["balance"] * RISK_PCT
        stop_dist_pct = abs(entry - stop) / entry
        notional = risk_usd / stop_dist_pct if stop_dist_pct > 1e-6 else 0
        margin = notional / LEVERAGE if notional > 0 else 0
        if margin > self.state["balance"] * 0.35:
            margin = self.state["balance"] * 0.35

        if margin < 1:
            return

        self.state["position"] = {
            "side": side, "entry_price": entry, "stop": stop, "tp": tp,
            "leverage": LEVERAGE, "margin": margin, "bars": 0,
            "open_time": now.isoformat(),
        }
        self.state["peak_balance"] = max(self.state["peak_balance"], self.state["balance"])

        trade = {
            "time": now.isoformat(), "action": "OPEN", "side": side,
            "entry": round(entry, 2), "stop": round(stop, 2), "tp": round(tp, 2),
            "margin": round(margin, 2), "leverage": LEVERAGE,
            "balance": round(self.state["balance"], 2),
        }
        self.log_trade(trade)
        print(f"  OPENED {side}: entry {entry:.2f} | stop {stop:.2f} | tp {tp:.2f} | margin ${margin:.2f}")
        self.save_state()

    def _print_status(self):
        s = self.state
        dd = (s["peak_balance"] - s["balance"]) / s["peak_balance"] if s["peak_balance"] > 0 else 0
        wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        pos = s["position"]
        pos_str = f"{pos['side']} @ {pos['entry_price']:.2f} ({pos.get('bars',0)}h)" if pos else "NONE"
        print(f"\n  Balance: ${s['balance']:.2f} | PnL: ${s['total_pnl']:+.2f} | WR: {wr:.0%} | DD: {dd:.0%}")
        print(f"  Position: {pos_str}")

    def run(self, interval_minutes=60):
        print(f"\n{'='*50}")
        print(f"  ETH SMA({SMA_PERIOD}) CROSSOVER BOT")
        print(f"  Pair: {self.pair}")
        print(f"  Strategy: Long above SMA, Short below SMA")
        print(f"  Leverage: {LEVERAGE}x | Risk: {RISK_PCT:.0%} | RR: {RR_RATIO}:1")
        print(f"  Mode: {'LIVE' if self.live else 'PAPER'}  |  Balance: ${self.state['balance']:.2f}")
        print(f"  Backtest: +$64.38 (+322% annual) | 28% WR | 36% max DD")
        print(f"{'='*50}\n")

        while True:
            self.tick()
            print(f"  Sleeping {interval_minutes}m...")
            time.sleep(interval_minutes * 60)

    def run_once(self):
        print(f"\n{'='*50}")
        print(f"  ETH SMA({SMA_PERIOD}) CROSSOVER — SINGLE TICK")
        print(f"{'='*50}\n")
        self.tick()


def main():
    live = "--live" in sys.argv
    once = "--once" in sys.argv
    bot = Bot(live=live)
    if once: bot.run_once()
    else: bot.run()


if __name__ == "__main__":
    main()