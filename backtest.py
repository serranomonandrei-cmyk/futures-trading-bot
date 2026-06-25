"""
Backtest engine. Event-driven, bar-by-bar. No look-ahead bias.
Signal on bar N close -> execute on bar N+1 open.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from config import (
    STARTING_BALANCE, TAKER_FEE_PCT, SLIPPAGE_PCT, FUNDING_RATE_PCT,
    MAX_LEVERAGE, MAX_RISK_PCT, MAX_CONCURRENT_POSITIONS,
    MIN_STOP_DISTANCE_PCT, DONCHIAN_RR_RATIO
)


@dataclass
class Trade:
    pair: str
    strategy: str
    side: str          # "long" or "short"
    entry_price: float
    stop: float
    tp: float
    leverage: int
    margin: float
    risk_usd: float
    entry_regime: str = "unknown"
    entry_time: object = None
    exit_price: float = 0.0
    exit_time: object = None
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    status: str = "open"
    bars_held: int = 0


@dataclass
class Portfolio:
    balance: float = STARTING_BALANCE
    peak_balance: float = STARTING_BALANCE
    trades: list = field(default_factory=list)
    open_positions: list = field(default_factory=list)

    @property
    def drawdown(self):
        return (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0

    @property
    def margin_used(self):
        return sum(p.margin for p in self.open_positions)

    @property
    def margin_utilization(self):
        return self.margin_used / self.balance if self.balance > 0 else 0


def calc_stop(side, entry, atr_val, atr_mult):
    """Calculate stop distance. Min 0.5% from entry."""
    raw_dist = atr_val * atr_mult
    min_dist = entry * MIN_STOP_DISTANCE_PCT
    dist = max(raw_dist, min_dist)
    if side == "long":
        return entry - dist
    return entry + dist


def calc_tp(side, entry, stop, rr_ratio=None):
    """Take profit at R:R ratio."""
    if rr_ratio is None:
        rr_ratio = DONCHIAN_RR_RATIO
    risk = abs(entry - stop)
    if side == "long":
        return entry + risk * rr_ratio
    return entry - risk * rr_ratio


def calc_position_size(balance, risk_pct, entry, stop, leverage):
    """Risk-based position sizing."""
    risk_usd = balance * risk_pct
    stop_dist_pct = abs(entry - stop) / entry
    if stop_dist_pct < 1e-6:
        return 0, 0, 0
    notional = risk_usd / stop_dist_pct
    margin = notional / leverage
    # Cap margin at 35% of balance
    if margin > balance * 0.35:
        margin = balance * 0.35
        notional = margin * leverage
        risk_usd = notional * stop_dist_pct
    return notional, margin, risk_usd


def apply_slippage(price, side, slippage_pct=SLIPPAGE_PCT):
    """Worse fill in live."""
    if side == "long":
        return price * (1 + slippage_pct)
    return price * (1 - slippage_pct)


class Backtester:
    def __init__(self, pair, strategy_name, strategy_fn, regime_col="regime",
                 leverage_map=None, risk_map=None, atr_stop_map=None):
        self.pair = pair
        self.strategy_name = strategy_name
        self.strategy_fn = strategy_fn
        self.regime_col = regime_col

        # Default regime-specific params
        self.leverage_map = leverage_map or {
            "trending_up": 15, "trending_down": 15,
            "ranging": 10, "volatile": 5, "unknown": 5
        }
        self.risk_map = risk_map or {
            "trending_up": 0.03, "trending_down": 0.03,
            "ranging": 0.02, "volatile": 0.015, "unknown": 0.015
        }
        self.atr_stop_map = atr_stop_map or {
            "trending_up": 2.0, "trending_down": 2.0,
            "ranging": 1.5, "volatile": 2.5, "unknown": 2.5
        }

    def run(self, df):
        """
        Run backtest on DataFrame with OHLCV + regime columns.
        Returns: Portfolio, metrics dict.
        """
        portfolio = Portfolio()

        # Generate signals (bar N close)
        signals = self.strategy_fn(df)

        # Need ATR for stop calculation
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        df = df.copy()
        df["atr"] = tr.rolling(14).mean()

        # === MAIN LOOP ===
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            signal = signals.iloc[i-1]  # Signal from PREVIOUS bar's close

            # 1. Check existing positions (SL/TP on current bar's high/low)
            self._check_positions(portfolio, df, i)

            # 2. Execute signal from previous bar (open of current bar)
            if signal != 0 and len(portfolio.open_positions) < MAX_CONCURRENT_POSITIONS:
                self._execute_entry(portfolio, df, i, signal)

            # 3. Track peak balance
            portfolio.peak_balance = max(portfolio.peak_balance, portfolio.balance)

        # Force close remaining positions at last bar close
        self._close_all(portfolio, df, len(df)-1)

        metrics = self._calc_metrics(portfolio, df)
        return portfolio, metrics

    def _execute_entry(self, portfolio, df, bar_idx, signal):
        """Enter position at bar open (after signal from prev bar close)."""
        bar = df.iloc[bar_idx]
        regime = bar.get(self.regime_col, "unknown")
        atr_val = bar.get("atr", 0)
        if pd.isna(atr_val) or atr_val <= 0:
            return

        side = "long" if signal == 1 else "short"
        entry_raw = bar["open"]
        entry = apply_slippage(entry_raw, side)

        leverage = self.leverage_map.get(regime, 10)
        risk_pct = self.risk_map.get(regime, 0.02)
        atr_mult = self.atr_stop_map.get(regime, 2.0)

        stop = calc_stop(side, entry, atr_val, atr_mult)
        tp = calc_tp(side, entry, stop)
        notional, margin, risk_usd = calc_position_size(
            portfolio.balance, risk_pct, entry, stop, leverage
        )

        if margin < 1 or portfolio.margin_used + margin > portfolio.balance * 0.75:
            return

        trade = Trade(
            pair=self.pair, strategy=self.strategy_name, side=side,
            entry_price=entry, stop=stop, tp=tp, leverage=leverage,
            margin=margin, risk_usd=risk_usd, entry_regime=regime,
            entry_time=bar["timestamp"] if "timestamp" in bar.index else bar_idx
        )
        portfolio.open_positions.append(trade)

    def _check_positions(self, portfolio, df, bar_idx):
        """Check SL/TP and regime change exit."""
        bar = df.iloc[bar_idx]
        current_regime = bar.get(self.regime_col, "unknown")
        closed = []

        for pos in portfolio.open_positions:
            pos.bars_held += 1

            # Funding every 8 bars (8H on 1H timeframe)
            is_8h = pos.bars_held % 8 == 0
            if is_8h:
                pos.funding += pos.margin * pos.leverage * FUNDING_RATE_PCT

            # Regime change: close if market regime no longer matches entry regime
            regime_changed = (current_regime != pos.entry_regime and
                              pos.entry_regime != "unknown")

            if pos.side == "long":
                if bar["low"] <= pos.stop:
                    self._close_position(portfolio, pos, pos.stop, bar, "stop")
                    closed.append(pos)
                elif bar["high"] >= pos.tp:
                    self._close_position(portfolio, pos, pos.tp, bar, "tp")
                    closed.append(pos)
                elif regime_changed:
                    self._close_position(portfolio, pos, bar["close"], bar, "regime_change")
                    closed.append(pos)
            else:  # short
                if bar["high"] >= pos.stop:
                    self._close_position(portfolio, pos, pos.stop, bar, "stop")
                    closed.append(pos)
                elif bar["low"] <= pos.tp:
                    self._close_position(portfolio, pos, pos.tp, bar, "tp")
                    closed.append(pos)
                elif regime_changed:
                    self._close_position(portfolio, pos, bar["close"], bar, "regime_change")
                    closed.append(pos)

        for p in closed:
            if p in portfolio.open_positions:
                portfolio.open_positions.remove(p)

    def _close_position(self, portfolio, pos, exit_price, bar, reason):
        """Close position and calculate P&L."""
        exit_price = apply_slippage(exit_price, "short" if pos.side == "long" else "long")

        if pos.side == "long":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        gross_pnl = pnl_pct * pos.margin * pos.leverage
        fee_open = pos.margin * pos.leverage * TAKER_FEE_PCT
        fee_close = pos.margin * pos.leverage * TAKER_FEE_PCT
        total_fees = fee_open + fee_close

        pos.exit_price = exit_price
        pos.exit_time = bar["timestamp"] if "timestamp" in bar.index else 0
        pos.pnl = gross_pnl - total_fees - pos.funding
        pos.fees = total_fees
        pos.status = "closed"

        portfolio.balance += pos.pnl
        portfolio.trades.append(pos)

    def _close_all(self, portfolio, df, bar_idx):
        """Force close all open positions at last bar close."""
        bar = df.iloc[bar_idx]
        for pos in list(portfolio.open_positions):
            self._close_position(portfolio, pos, bar["close"], bar, "force_close")
        portfolio.open_positions.clear()

    def _calc_metrics(self, portfolio, df):
        """Calculate performance metrics."""
        trades = portfolio.trades
        if not trades:
            return {"error": "no trades"}

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Equity curve for drawdown
        equity = [STARTING_BALANCE]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        equity = np.array(equity)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        max_dd = drawdowns.max()

        # Sharpe (approximate, using trade returns, annualized for 1H data)
        if len(pnls) > 1:
            returns = np.array(pnls) / STARTING_BALANCE
            # sqrt(8760) = annualization factor for 1H bars (8760 hours/year)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(8760)
        else:
            sharpe = 0

        total_pnl = sum(pnls)
        total_fees = sum(t.fees for t in trades)
        total_funding = sum(t.funding for t in trades)

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_funding": total_funding,
            "net_pnl": total_pnl,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": sum(wins) / (abs(sum(losses)) + 1e-10),
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "final_balance": portfolio.balance,
            "return_pct": (portfolio.balance - STARTING_BALANCE) / STARTING_BALANCE * 100,
            "avg_bars_held": np.mean([t.bars_held for t in trades]),
            "leverage_used": np.mean([t.leverage for t in trades]),
        }
