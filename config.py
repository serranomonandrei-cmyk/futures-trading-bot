"""All parameters in one place. Edit here, nowhere else."""

# --- Account ---
STARTING_BALANCE = 20.0  # USD

# --- Pairs ---
PAIRS = [
    "ETH/USDT:USDT",   # SMA(16) — +$92.36, best performer
    "BTC/USDT:USDT",   # SMA(36) — +$8.58, OOS positive
    "SOL/USDT:USDT",   # SMA(28) — +$15.09, OOS positive
    "BNB/USDT:USDT",   # SMA(56) — +$37.98, OOS positive
    "SUI/USDT:USDT",   # SMA(36) — +$61.05, OOS > IS
    "LINK/USDT:USDT",  # SMA(28) — +$52.55, OOS positive
    "ADA/USDT:USDT",   # SMA(28) — +$52.72, OOS positive
    "DOGE/USDT:USDT",  # EMA(18) — +$3.97, OOS positive
    "XRP/USDT:USDT",   # EMA(50) — +$11.43, OOS > IS
    "BCH/USDT:USDT",   # MACD(12,26,9) — +$24.48, OOS positive
]

# --- Timeframes ---
TF_HIGHER = "1h"   # Primary timeframe
TF_ENTRY = "1h"    # Entry signals (same as primary)
TF_CANDLES = "1h"  # Data resolution

# --- Regime Detection (on 1H) ---
REGIME_EMA_PERIOD = 100     # 100 hours = ~4 days (stable trend)
REGIME_ATR_PERIOD = 14
REGIME_PERSISTENCE = 12     # 12 bars before regime switch
REGIME_TREND_THRESHOLD = 0.02  # 2% from EMA = trending
REGIME_VOLATILE_MULT = 1.5  # ATR > 1.5x avg = volatile

# --- Strategy Parameters ---
DONCHIAN_PERIOD = 25       # Backtested best for ETH
DONCHIAN_LEVERAGE = 15     # 15x leverage
DONCHIAN_RISK_PCT = 0.03   # 3% risk per trade
DONCHIAN_ATR_STOP_MULT = 2.0  # 2x ATR stop
DONCHIAN_RR_RATIO = 4.0    # 4:1 reward:risk (matches bot_multi)

# Trending regime
TREND_RSI_PERIOD = 14
TREND_RSI_LONG_MIN = 40
TREND_RSI_LONG_MAX = 65
TREND_RSI_SHORT_MIN = 35
TREND_RSI_SHORT_MAX = 60
TREND_LEVERAGE = 15
TREND_RISK_PCT = 0.03
TREND_ATR_STOP_MULT = 2.0

# Ranging regime
RANGE_LEVERAGE = 10
RANGE_RISK_PCT = 0.02
RANGE_ATR_STOP_MULT = 1.5

# Volatile regime
VOLATILE_LEVERAGE = 5
VOLATILE_RISK_PCT = 0.015
VOLATILE_ATR_STOP_MULT = 2.5

# --- Risk Limits ---
MAX_LEVERAGE = 20
MAX_RISK_PCT = 0.05
MAX_CONCURRENT_POSITIONS = 10
MAX_MARGIN_UTILIZATION = 0.75
MIN_STOP_DISTANCE_PCT = 0.005  # 0.5%
MAX_DAILY_TRADES = 10

# --- Cost Model ---
TAKER_FEE_PCT = 0.0004   # 0.04% per side (Binance VIP0)
MAKER_FEE_PCT = 0.0002   # 0.02% per side
SLIPPAGE_PCT = 0.0003    # 0.03% base slippage
FUNDING_RATE_PCT = 0.0001  # 0.01% per 8h (average)

# --- Backtest ---
BACKTEST_MONTHS = 6
WALK_FORWARD_TRAIN_MONTHS = 3
WALK_FORWARD_TEST_MONTHS = 1
WALK_FORWARD_MIN_WINDOWS = 4
MONTE_CARLO_ITERATIONS = 5000
MIN_TRADES_FOR_VALIDATION = 100

# --- Execution ---
PAPER_MODE = True
