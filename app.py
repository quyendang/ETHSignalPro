# ===============================
# crypto.py — FastAPI Crypto Bot (RSI, Symbol Tracker, Big Trades)
# ===============================
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import asyncpg
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Query, Request, Form, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# ------------------------------------------------------------------
# 1) GLOBAL APP/ENV CONFIG
# ------------------------------------------------------------------
app = FastAPI(title="Crypto Bot API")

# Templates
templates = Jinja2Templates(directory="templates")

def comma_format(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return value

templates.env.filters["comma"] = comma_format

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)

# PostgreSQL ENV (Koyeb)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://koyeb-adm:*******@ep-aged-bush-a1op42oy.ap-southeast-1.pg.koyeb.app/koyebdb")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL phải được thiết lập trong biến môi trường.")

# Parse DATABASE_URL
try:
    parsed = urlparse(DATABASE_URL)
    db_config = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "database": parsed.path.lstrip("/"),
    }
    
    # Validate required fields
    if not all([db_config["host"], db_config["user"], db_config["password"], db_config["database"]]):
        raise ValueError("DATABASE_URL thiếu thông tin cần thiết (host, user, password, database)")
except Exception as e:
    raise ValueError(f"Lỗi parse DATABASE_URL: {e}")

# PostgreSQL connection pool
_db_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    """Get or create database connection pool"""
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            min_size=2,
            max_size=10,
        )
    return _db_pool

async def init_db():
    """Initialize database tables if they don't exist"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Create bot_subscriptions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_subscriptions (
                symbol VARCHAR(20) PRIMARY KEY,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create eth_cycles table (if needed)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS eth_cycles (
                id SERIAL PRIMARY KEY,
                cycle_index INTEGER NOT NULL,
                buy_price NUMERIC,
                sell_price NUMERIC,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create index on cycle_index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_eth_cycles_cycle_index 
            ON eth_cycles(cycle_index DESC)
        """)
        logging.info("[DB] Database tables initialized")

# ------------------------------------------------------------------
# 2) RSI BOT CONFIG
# ------------------------------------------------------------------
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "")
RSI_SYMBOLS = [s.strip() for s in os.getenv("RSI_SYMBOLS", "ETHUSDT,BTCUSDT").split(",") if s.strip()] or ["ETHUSDT", "BTCUSDT"]
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_CHECK_MINUTES = int(os.getenv("RSI_CHECK_MINUTES", "5"))
RSI_TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d"}

# ETH TRACKER CONFIG
ETH_TRACKER_SYMBOL = os.getenv("ETH_TRACKER_SYMBOL", "ETHUSDT")
ETH_TRACKER_INTERVAL = os.getenv("ETH_TRACKER_INTERVAL", "4h")
ETH_CYCLE_SIZE = float(os.getenv("ETH_CYCLE_SIZE", "40"))
ETH_BASE_BALANCE = float(os.getenv("ETH_BASE_BALANCE", "138"))

BIG_ORDER_THRESHOLD = 100_000
INTERVAL_MS_MAP = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
TRACKER_INTERVAL = ETH_TRACKER_INTERVAL

ETH_SELL_ZONE_LOW = float(os.getenv("ETH_SELL_ZONE_LOW", "3650"))
ETH_SELL_ZONE_HIGH = float(os.getenv("ETH_SELL_ZONE_HIGH", "3700"))
ETH_BUY_ZONE_LOW = float(os.getenv("ETH_BUY_ZONE_LOW", "3350"))
ETH_BUY_ZONE_HIGH = float(os.getenv("ETH_BUY_ZONE_HIGH", "3450"))
ETH_RSI_SELL = float(os.getenv("ETH_RSI_SELL", "65"))
ETH_RSI_BUY = float(os.getenv("ETH_RSI_BUY", "40"))

MACD_FAST = int(os.getenv("ETH_MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("ETH_MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("ETH_MACD_SIGNAL", "9"))

# State
_rsi_last_state: Dict[str, Dict[str, str]] = {sym: {tf: "unknown" for tf in RSI_TIMEFRAMES} for sym in RSI_SYMBOLS}
_rsi_last_values: Dict[str, Dict[str, Dict[str, float]]] = {}
_rsi_last_run: float = 0.0

# Router
_rsi_router = APIRouter()

# ------------------------------------------------------------------
# 3) CRYPTO FUNCTIONS
# ------------------------------------------------------------------

def _rsi_wilder(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        raise ValueError("Not enough data to compute RSI")
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _rsi_fetch_klines(symbol: str, interval: str, limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def _compute_eth_zones_from_range(symbol: str, interval: str, lookback: int = 60):
    """
    Tính vùng BUY/SELL zone dựa trên high/low của N cây H4 gần nhất.
    """
    kl = _rsi_fetch_klines(symbol, interval, limit=lookback)
    if len(kl) < lookback:
        raise ValueError("Not enough klines for dynamic zone calc")

    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]

    recent_high = max(highs)
    recent_low = min(lows)
    price_range = recent_high - recent_low

    if price_range <= 0:
        raise ValueError("Invalid price range")

    zone_pct = 0.2

    buy_low = recent_low
    buy_high = recent_low + zone_pct * price_range

    sell_high = recent_high
    sell_low = recent_high - zone_pct * price_range

    return sell_low, sell_high, buy_low, buy_high, recent_low, recent_high


def _compute_ema_series(values: List[float], period: int) -> List[Optional[float]]:
    """
    Trả về list EMA cùng độ dài với values.
    """
    if len(values) < period:
        raise ValueError(f"Not enough data for EMA({period})")

    ema_values: List[Optional[float]] = [None] * len(values)
    sma = sum(values[:period]) / period
    ema_values[period - 1] = sma

    k = 2 / (period + 1)
    ema_prev = sma
    for i in range(period, len(values)):
        ema = (values[i] - ema_prev) * k + ema_prev
        ema_values[i] = ema
        ema_prev = ema

    return ema_values


def _macd_latest(symbol: str, interval: str, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    """
    Tính MACD (fast, slow, signal) cho symbol/interval.
    """
    limit = max(200, slow * 5)
    kl = _rsi_fetch_klines(symbol, interval, limit=limit)
    closes = [float(k[4]) for k in kl]

    if len(closes) < slow + signal + 5:
        raise ValueError("Not enough data to compute MACD")

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: List[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    if signal_line is None:
        raise ValueError("Signal line not ready")

    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _macd_latest_with_prev(
    symbol: str,
    interval: str,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """
    Tính MACD cho symbol/interval.
    Trả về (macd_line, signal_line, hist, prev_hist):
    """
    limit = max(200, slow * 5)
    kl = _rsi_fetch_klines(symbol, interval, limit=limit)
    closes = [float(k[4]) for k in kl]

    if len(closes) < slow + signal + 5:
        raise ValueError("Not enough data to compute MACD")

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: List[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    prev_signal_line = signal_series[-2]

    if signal_line is None or prev_signal_line is None:
        raise ValueError("Signal line not ready")

    hist = macd_line - signal_line
    prev_hist = macd_series[-2] - prev_signal_line

    return macd_line, signal_line, hist, prev_hist


def _rsi_latest(symbol: str, interval: str, period: int):
    kl = _rsi_fetch_klines(symbol, interval, limit=max(200, period * 5))
    closes = [float(k[4]) for k in kl]
    rsi = _rsi_wilder(closes, period=period)
    price = closes[-1]
    return price, rsi


def _pushover_notify(title: str, message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return
    data = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": 0,
        "sound": "cash",
    }
    if PUSHOVER_DEVICE:
        data["device"] = PUSHOVER_DEVICE
    try:
        requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=15)
    except Exception:
        pass


def _fmt_dual(tf: str, condition: str, snapshot: Dict[str, Dict[str, float]]):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "Z"
    lines = [f"TF: {tf} | Cond: {condition} | RSI({RSI_PERIOD}) | {ts}"]
    ordered = sorted(snapshot.items(), key=lambda kv: (0 if kv[0].upper() == "ETHUSDT" else 1, kv[0]))
    for sym, v in ordered:
        if "price" in v and "rsi" in v:
            lines.append(f"{sym}: Price {v['price']:.2f} | RSI {v['rsi']:.2f}")
        else:
            lines.append(f"{sym}: error {v.get('error', 'unknown')}")
    return "\n".join(lines)


def _compute_rsi_series(closes: list[float], period: int) -> list[float]:
    """
    Tính RSI series classic từ list closes.
    """
    if len(closes) < period + 2:
        return [50.0] * len(closes)

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    def ema(series, p):
        alpha = 2 / (p + 1)
        ema_vals = []
        prev = sum(series[:p]) / p
        ema_vals.append(prev)
        for v in series[p:]:
            prev = alpha * v + (1 - alpha) * prev
            ema_vals.append(prev)
        return ema_vals

    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)

    rsi = [50.0] * len(closes)
    offset = len(closes) - len(avg_gain)
    for i in range(len(avg_gain)):
        if avg_loss[i] == 0:
            rs = float('inf')
            r = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            r = 100 - (100 / (1 + rs))
        rsi[offset + i] = r

    return rsi


def _compute_macd_series(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """
    Tính MACD series cho 1 list closes.
    Trả về (macd_line[], signal_line[], hist[])
    """
    if len(closes) < slow + signal + 5:
        n = len(closes)
        return [0.0]*n, [0.0]*n, [0.0]*n

    ema_fast = _compute_ema_series(closes, fast)
    ema_slow = _compute_ema_series(closes, slow)

    macd_series: list[float] = []
    for ef, es in zip(ema_fast, ema_slow):
        if ef is None or es is None:
            macd_series.append(0.0)
        else:
            macd_series.append(ef - es)

    signal_series = _compute_ema_series(macd_series, signal)
    hist_series: list[float] = []
    for m, s in zip(macd_series, signal_series):
        if s is None:
            hist_series.append(0.0)
        else:
            hist_series.append(m - s)

    return macd_series, signal_series, hist_series

def _sma_series(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    if n < period:
        return [None] * n
    out: list[float | None] = [None] * (period - 1)
    window_sum = sum(values[:period])
    out.append(window_sum / period)
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out.append(window_sum / period)
    return out


def _bollinger_bands(values: list[float], period: int = 20, k: float = 2.0):
    """
    Trả về (middle[], upper[], lower[])
    """
    n = len(values)
    middle = _sma_series(values, period)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n

    if n < period:
        return middle, upper, lower

    import math

    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        m = middle[i]
        if m is None:
            continue
        variance = sum((v - m) ** 2 for v in window) / period
        std = math.sqrt(variance)
        upper[i] = m + k * std
        lower[i] = m - k * std

    return middle, upper, lower


def _stochastic_oscillator(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """
    %K: 0..100
    """
    n = len(closes)
    if n < period:
        return [None] * n

    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window_high = max(highs[i - period + 1 : i + 1])
        window_low = min(lows[i - period + 1 : i + 1])
        if window_high == window_low:
            out[i] = 50.0
        else:
            out[i] = (closes[i] - window_low) / (window_high - window_low) * 100.0
    return out


def _williams_r(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """
    Williams %R: -100 .. 0
    """
    n = len(closes)
    if n < period:
        return [None] * n

    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window_high = max(highs[i - period + 1 : i + 1])
        window_low = min(lows[i - period + 1 : i + 1])
        if window_high == window_low:
            out[i] = -50.0
        else:
            out[i] = -100.0 * (window_high - closes[i]) / (window_high - window_low)
    return out


def _rsi_check_once():
    global _rsi_last_state, _rsi_last_values, _rsi_last_run
    snap_all: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tf, interval in RSI_TIMEFRAMES.items():
        tf_snap: Dict[str, Dict[str, float]] = {}

        for sym in RSI_SYMBOLS:
            try:
                price, rsi = _rsi_latest(sym, interval, RSI_PERIOD)
                tf_snap[sym] = {"price": price, "rsi": rsi}
            except Exception as e:
                tf_snap[sym] = {"error": str(e)}

        for sym in RSI_SYMBOLS:
            v = tf_snap.get(sym, {})
            rsi = v.get("rsi")
            if rsi is None:
                continue
            prev = _rsi_last_state.get(sym, {}).get(tf, "unknown")
            if rsi < 30 and prev != "oversold":
                _pushover_notify(f"RSI Oversold {tf} — {sym}", _fmt_dual(tf, "<30", tf_snap))
                _rsi_last_state[sym][tf] = "oversold"
            elif rsi > 70 and prev != "overbought":
                _pushover_notify(f"RSI Overbought {tf} — {sym}", _fmt_dual(tf, ">70", tf_snap))
                _rsi_last_state[sym][tf] = "overbought"
            elif 30 <= rsi <= 70 and prev != "normal":
                _rsi_last_state[sym][tf] = "normal"

        snap_all[tf] = tf_snap

    _rsi_last_values = snap_all
    _rsi_last_run = time.time()
    return snap_all
    
def _eth_decide_action(
    price: float,
    rsi_h4: float,
    macd_hist: float,
    prev_macd_hist: float,
    zones: tuple,
    btc_rsi_h4: float,
    btc_macd_hist: float,
    btc_prev_macd_hist: float,
) -> Dict[str, str]:
    """
    Quyết định BUY/SELL/HOLD với BTC filter.
    """
    sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones

    reasons: List[str] = []
    reasons.append(
        f"Dynamic zones: BUY[{buy_low:.1f}-{buy_high:.1f}] "
        f"SELL[{sell_low:.1f}-{sell_high:.1f}] "
        f"(range {recent_low:.1f}-{recent_high:.1f})"
    )

    action = "HOLD"

    macd_weakening = (
        macd_hist > 0
        and prev_macd_hist is not None
        and macd_hist < prev_macd_hist
    )

    if (
        sell_low <= price <= sell_high
        and rsi_h4 >= ETH_RSI_SELL
        and macd_weakening
    ):
        action = "SELL"
        reasons.append(
            f"Price {price:.1f} in SELL zone & RSI_H4 {rsi_h4:.1f} >= {ETH_RSI_SELL}"
        )
        reasons.append(
            f"MACD hist weakening: current {macd_hist:.4f} < prev {prev_macd_hist:.4f}"
        )

    elif buy_low <= price <= buy_high and rsi_h4 <= ETH_RSI_BUY:
        action = "BUY"
        reasons.append(
            f"Price {price:.1f} in BUY zone & RSI_H4 {rsi_h4:.1f} <= {ETH_RSI_BUY}"
        )

    else:
        reasons.append("No buy/sell condition matched (HOLD).")

    btc_bull_rsi = btc_rsi_h4 >= 65
    btc_macd_stronger = (
        btc_macd_hist > 0
        and btc_prev_macd_hist is not None
        and btc_macd_hist >= btc_prev_macd_hist
    )

    if action == "SELL" and (btc_bull_rsi or btc_macd_stronger):
        reasons.append(
            f"Cancel SELL: BTC still bullish (RSI_H4={btc_rsi_h4:.1f}, "
            f"MACD hist {btc_macd_hist:.4f} >= prev {btc_prev_macd_hist:.4f})"
        )
        action = "HOLD"

    if abs(macd_hist) < 0.5:
        reasons.append("MACD hist ~0 → momentum weak / sideway.")
    elif macd_hist > 0:
        reasons.append("MACD hist > 0 → bullish momentum.")
    else:
        reasons.append("MACD hist < 0 → bearish momentum.")

    return {
        "action": action,
        "reason": " | ".join(reasons),
    }


async def _get_next_cycle_index() -> int:
    """Get next cycle index from database"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT cycle_index FROM eth_cycles ORDER BY cycle_index DESC LIMIT 1"
            )
            if row:
                return int(row["cycle_index"]) + 1
            return 1
    except Exception as e:
        logging.error(f"[DB] Error getting next cycle index: {e}")
        return 1


async def _get_open_cycle():
    """Get open cycle from database"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM eth_cycles WHERE buy_price IS NULL ORDER BY cycle_index DESC LIMIT 1"
            )
            if row:
                return dict(row)
            return None
    except Exception as e:
        logging.error(f"[DB] Error getting open cycle: {e}")
        return None


def run_symbol_tracker_once(symbol: str, send_notify: bool = False) -> Dict[str, Any]:
    """
    Tracker chung cho mọi symbol.
    """
    symbol = symbol.upper()
    interval = TRACKER_INTERVAL

    price, rsi_h4 = _rsi_latest(symbol, interval, RSI_PERIOD)

    macd_line, macd_signal, macd_hist, prev_macd_hist = _macd_latest_with_prev(
        symbol,
        interval,
    )

    btc_price, btc_rsi_h4 = _rsi_latest("BTCUSDT", interval, RSI_PERIOD)

    _, _, btc_macd_hist, btc_prev_macd_hist = _macd_latest_with_prev(
        "BTCUSDT",
        interval,
    )

    zones = _compute_eth_zones_from_range(symbol, interval, lookback=60)
    sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones

    decision = _eth_decide_action(
        price=price,
        rsi_h4=rsi_h4,
        macd_hist=macd_hist,
        prev_macd_hist=prev_macd_hist,
        zones=zones,
        btc_rsi_h4=btc_rsi_h4,
        btc_macd_hist=btc_macd_hist,
        btc_prev_macd_hist=btc_prev_macd_hist,
    )
    action = decision["action"]
    reason = decision["reason"]

    now_utc = datetime.utcnow().isoformat() + "Z"

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": interval,
        "now_utc": now_utc,
        "price": price,
        "rsi_h4": rsi_h4,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "action": action,
        "reason": reason,
        "zones": {
            "sell_low": sell_low,
            "sell_high": sell_high,
            "buy_low": buy_low,
            "buy_high": buy_high,
            "recent_low": recent_low,
            "recent_high": recent_high,
        },
        "btc": {
            "price": btc_price,
            "rsi_h4": btc_rsi_h4,
            "macd_hist": btc_macd_hist,
            "prev_macd_hist": btc_prev_macd_hist,
        },
    }

    if send_notify and action != "HOLD":
        try:
            title = f"[{action}] {symbol} 💰"
            msg_lines = [
                f"Price: {price}",
                f"Reason: {reason}",
                f"RSI H4: {rsi_h4:.2f}",
                f"MACD: {macd_line:.4f} | Signal: {macd_signal:.4f} | Hist: {macd_hist:.4f}",
                f"BTC RSI H4: {btc_rsi_h4:.1f}, BTC hist: {btc_macd_hist:.4f}",
                f"Time (UTC): {now_utc}",
            ]
            _pushover_notify(title, "\n".join(msg_lines))
        except Exception as e:
            logging.error(f"[SYMBOL_TRACKER_NOTIFY] Error: {e}")

    return payload


async def symbols_tracker_job():
    """
    Job chạy mỗi 10 phút.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT symbol FROM bot_subscriptions WHERE is_active = TRUE"
            )
    except Exception as e:
        logging.error(f"[SYMBOL_TRACKER_JOB] Error fetch subscriptions: {e}")
        return

    for row in rows:
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            payload = run_symbol_tracker_once(symbol, send_notify=True)
            logging.info(
                f"[SYMBOL_TRACKER_JOB] {symbol}: action={payload['action']} price={payload['price']}"
            )
        except Exception as e:
            logging.error(f"[SYMBOL_TRACKER_JOB] {symbol}: error {e}")


def _run_async_job():
    """Wrapper to run async job in sync context for scheduler"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(symbols_tracker_job())

def init_inline_rsi_dual(app_: FastAPI, scheduler: Optional[BackgroundScheduler] = None):
    app_.include_router(_rsi_router, prefix="/bots", tags=["bots"])
    if scheduler is not None:
        try:
            scheduler.add_job(
                _run_async_job,
                "interval",
                minutes=10,
                id="symbols_tracker_job",
                replace_existing=True,
                next_run_time=datetime.utcnow(),
            )
        except Exception:
            scheduler.add_job(
                _run_async_job,
                "interval",
                minutes=10,
                id="symbols_tracker_job",
                replace_existing=True,
            )
    else:
        import threading

        def _loop():
            while True:
                try:
                    _rsi_check_once()
                except Exception:
                    pass
                time.sleep(RSI_CHECK_MINUTES * 60)

        threading.Thread(target=_loop, daemon=True).start()

# ------------------------------------------------------------------
# 4) CRYPTO ENDPOINTS
# ------------------------------------------------------------------

@_rsi_router.post("/subscribe")
async def subscribe_symbol(symbol: str = Form(...)):
    """
    SUBSCRIBE 1 symbol vào danh sách theo dõi.
    """
    symbol = symbol.upper()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO bot_subscriptions (symbol, is_active, updated_at)
                VALUES ($1, TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol) 
                DO UPDATE SET is_active = TRUE, updated_at = CURRENT_TIMESTAMP
            """, symbol)
    except Exception as e:
        logging.error(f"[SUBSCRIBE] Error subscribe {symbol}: {e}")

    return RedirectResponse(url=f"/bots/{symbol}", status_code=303)


@_rsi_router.post("/unsubscribe")
async def unsubscribe_symbol(symbol: str = Form(...)):
    """
    UNSUBSCRIBE 1 symbol khỏi danh sách theo dõi.
    """
    symbol = symbol.upper()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE bot_subscriptions 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = $1
            """, symbol)
    except Exception as e:
        logging.error(f"[UNSUBSCRIBE] Error unsubscribe {symbol}: {e}")

    return RedirectResponse(url=f"/bots/{symbol}", status_code=303)


@_rsi_router.get("/{symbol}", response_class=HTMLResponse)
async def symbol_dashboard(request: Request, symbol: str):
    """
    Dashboard theo dõi bất kỳ symbol nào.
    """
    symbol = symbol.upper()

    try:
        klines = _rsi_fetch_klines(symbol, TRACKER_INTERVAL, limit=200)
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error fetching klines for {symbol}: {e}")
        klines = []

    labels: List[str] = []
    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    for k in klines:
        try:
            open_time_ms = int(k[0])
            dt = datetime.utcfromtimestamp(open_time_ms / 1000.0)
            labels.append(dt.strftime("%Y-%m-%d %H:%M"))

            o = float(k[1])
            h = float(k[2])
            l = float(k[3])
            c = float(k[4])

            highs.append(h)
            lows.append(l)
            closes.append(c)
        except Exception as e:
            logging.warning(f"[SYMBOL DASH] Bad kline row for {symbol}: {e}")
            continue

    if not closes:
        context = {
            "request": request,
            "symbol": symbol,
            "rows_json": [],
            "last_price": None,
            "last_rsi": None,
            "change_24h": None,
            "buy_low": None,
            "buy_high": None,
            "sell_low": None,
            "sell_high": None,
            "is_subscribed": False,
            "tracker_action": "HOLD",
            "tracker_reason": "No data",
        }
        return templates.TemplateResponse("symbol_dashboard.html", context)

    rsi_values = _compute_rsi_series(closes, RSI_PERIOD)
    macd_line, macd_signal, macd_hist_values = _compute_macd_series(
        closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    ema_fast = _compute_ema_series(closes, 12)
    ema_slow = _compute_ema_series(closes, 26)
    sma_50 = _sma_series(closes, 50)

    bb_middle, bb_upper, bb_lower = _bollinger_bands(closes, period=20, k=2.0)
    stoch_k = _stochastic_oscillator(highs, lows, closes, period=14)
    williams_r = _williams_r(highs, lows, closes, period=14)

    buy_low = buy_high = sell_low = sell_high = recent_low = recent_high = None
    try:
        zones = _compute_eth_zones_from_range(symbol, TRACKER_INTERVAL, lookback=60)
        sell_low, sell_high, buy_low, buy_high, recent_low, recent_high = zones
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error computing zones for {symbol}: {e}")

    n = len(closes)
    min_len = min(
        n,
        len(labels),
        len(rsi_values),
        len(macd_hist_values),
        len(ema_fast),
        len(ema_slow),
        len(bb_upper),
        len(bb_lower),
        len(stoch_k),
        len(williams_r),
    )

    labels = labels[-min_len:]
    closes = closes[-min_len:]
    rsi_values = rsi_values[-min_len:]
    macd_hist_values = macd_hist_values[-min_len:]
    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    bb_upper = bb_upper[-min_len:]
    bb_lower = bb_lower[-min_len:]
    stoch_k = stoch_k[-min_len:]
    williams_r = williams_r[-min_len:]
    highs = highs[-min_len:]
    lows = lows[-min_len:]

    rows_json: List[Dict[str, Any]] = []
    for i in range(min_len):
        rows_json.append(
            {
                "time_str": labels[i],
                "price": closes[i],
                "rsi_h4": rsi_values[i],
                "macd_hist": macd_hist_values[i],
                "ema_fast": ema_fast[i],
                "ema_slow": ema_slow[i],
                "bb_upper": bb_upper[i],
                "bb_lower": bb_lower[i],
                "stoch_k": stoch_k[i],
                "wr": williams_r[i],
            }
        )

    last_price = closes[-1]
    last_rsi = rsi_values[-1] if rsi_values else None

    change_24h = None
    try:
        if len(closes) >= 7:
            ref = closes[-7]
            if ref != 0:
                change_24h = (last_price - ref) / ref * 100.0
    except Exception:
        change_24h = None

    tracker_action = "HOLD"
    tracker_reason = ""
    try:
        payload = run_symbol_tracker_once(symbol, send_notify=False)
        tracker_action = payload.get("action", "HOLD")
        tracker_reason = payload.get("reason", "")
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error run_symbol_tracker_once for {symbol}: {e}")

    is_subscribed = False
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT is_active FROM bot_subscriptions WHERE symbol = $1 LIMIT 1",
                symbol
            )
            if row and row.get("is_active"):
                is_subscribed = True
    except Exception as e:
        logging.error(f"[SYMBOL DASH] Error check subscription for {symbol}: {e}")

    context = {
        "request": request,
        "symbol": symbol,
        "rows_json": rows_json,
        "last_price": last_price,
        "last_rsi": last_rsi,
        "change_24h": change_24h,
        "buy_low": buy_low,
        "buy_high": buy_high,
        "sell_low": sell_low,
        "sell_high": sell_high,
        "is_subscribed": is_subscribed,
        "tracker_action": tracker_action,
        "tracker_reason": tracker_reason,
    }

    return templates.TemplateResponse("symbol_dashboard.html", context)


@_rsi_router.post("/backtest")
async def backtest_symbol(
    symbol: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    interval: str = Form("4h"),
    lookback: int = Form(60),
    mode: str = Form("history"),  # "history" or "realtime"
):
    """
    Backtest symbol với thuật toán buy/sell zone và signals.
    mode: "history" (dữ liệu lịch sử) hoặc "realtime" (dữ liệu realtime)
    """
    from fastapi.responses import JSONResponse
    
    symbol = symbol.upper()
    
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        
        if start_dt >= end_dt:
            return JSONResponse(
                {"error": "Start date must be before end date"},
                status_code=400
            )
        
        # Fetch klines từ Binance
        if mode == "history":
            # Lấy dữ liệu lịch sử
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            
            all_klines = []
            current_ts = start_ts
            limit = 1000  # Binance max limit
            
            while current_ts < end_ts:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": limit,
                }
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                batch = resp.json()
                
                if not batch:
                    break
                
                all_klines.extend(batch)
                current_ts = batch[-1][0] + 1  # Next start time
                
                if len(batch) < limit:
                    break
        else:  # realtime
            # Lấy dữ liệu realtime (từ hiện tại trở về trước)
            limit = int((end_dt - start_dt).total_seconds() / (4 * 3600)) + 100  # Approximate
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000),
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            all_klines = resp.json()
            
            # Filter theo date range
            filtered_klines = []
            for k in all_klines:
                k_time = datetime.utcfromtimestamp(int(k[0]) / 1000.0)
                if start_dt <= k_time <= end_dt:
                    filtered_klines.append(k)
            all_klines = filtered_klines
        
        if not all_klines:
            return JSONResponse(
                {"error": "No data found for the specified date range"},
                status_code=404
            )
        
        # Process klines
        closes = []
        highs = []
        lows = []
        timestamps = []
        
        for k in all_klines:
            try:
                ts_ms = int(k[0])
                dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
                
                if start_dt <= dt <= end_dt:
                    timestamps.append(dt.isoformat() + "Z")
                    closes.append(float(k[4]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
            except Exception as e:
                logging.warning(f"[BACKTEST] Bad kline: {e}")
                continue
        
        if len(closes) < lookback + 10:
            return JSONResponse(
                {"error": f"Not enough data (need at least {lookback + 10} candles)"},
                status_code=400
            )
        
        # Tính indicators
        rsi_values = _compute_rsi_series(closes, RSI_PERIOD)
        macd_line, macd_signal, macd_hist_values = _compute_macd_series(
            closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )
        ema_fast = _compute_ema_series(closes, 12)
        ema_slow = _compute_ema_series(closes, 26)
        bb_middle, bb_upper, bb_lower = _bollinger_bands(closes, period=20, k=2.0)
        stoch_k = _stochastic_oscillator(highs, lows, closes, period=14)
        williams_r = _williams_r(highs, lows, closes, period=14)
        
        # Fetch BTC data for the same period (for BTC filter)
        btc_klines = []
        try:
            if mode == "history":
                start_ts = int(start_dt.timestamp() * 1000)
                end_ts = int(end_dt.timestamp() * 1000)
                current_ts = start_ts
                limit = 1000
                
                while current_ts < end_ts:
                    url = "https://api.binance.com/api/v3/klines"
                    params = {
                        "symbol": "BTCUSDT",
                        "interval": interval,
                        "startTime": current_ts,
                        "endTime": end_ts,
                        "limit": limit,
                    }
                    resp = requests.get(url, params=params, timeout=15)
                    resp.raise_for_status()
                    batch = resp.json()
                    if not batch:
                        break
                    btc_klines.extend(batch)
                    current_ts = batch[-1][0] + 1
                    if len(batch) < limit:
                        break
            else:
                limit = int((end_dt - start_dt).total_seconds() / (4 * 3600)) + 100
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": min(limit, 1000),
                }
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                btc_klines = resp.json()
        except Exception as e:
            logging.warning(f"[BACKTEST] Error fetching BTC data: {e}")
            btc_klines = []
        
        # Process BTC klines
        btc_closes = []
        btc_timestamps = []
        for k in btc_klines:
            try:
                ts_ms = int(k[0])
                dt = datetime.utcfromtimestamp(ts_ms / 1000.0)
                if start_dt <= dt <= end_dt:
                    btc_timestamps.append(dt.isoformat() + "Z")
                    btc_closes.append(float(k[4]))
            except Exception:
                continue
        
        # Calculate BTC indicators
        btc_rsi_values = []
        btc_macd_hist_values = []
        if btc_closes and len(btc_closes) >= 50:
            btc_rsi_values = _compute_rsi_series(btc_closes, RSI_PERIOD)
            _, _, btc_macd_hist_values = _compute_macd_series(
                btc_closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
            )
        
        # Tính zones và signals cho mỗi điểm
        signals = []
        buy_count = 0
        sell_count = 0
        
        for i in range(lookback, len(closes)):
            try:
                # Tính zones dựa trên lookback candles trước đó
                window_highs = highs[i - lookback:i]
                window_lows = lows[i - lookback:i]
                
                if not window_highs or not window_lows:
                    continue
                
                recent_high = max(window_highs)
                recent_low = min(window_lows)
                price_range = recent_high - recent_low
                
                if price_range <= 0:
                    continue
                
                zone_pct = 0.2
                buy_low = recent_low
                buy_high = recent_low + zone_pct * price_range
                sell_high = recent_high
                sell_low = recent_high - zone_pct * price_range
                
                price = closes[i]
                rsi = rsi_values[i] if i < len(rsi_values) else 50.0
                macd_hist = macd_hist_values[i] if i < len(macd_hist_values) else 0.0
                prev_macd_hist = macd_hist_values[i - 1] if i > 0 and i - 1 < len(macd_hist_values) else None
                
                # Get BTC data for this point (find closest timestamp)
                btc_rsi_h4 = 50.0
                btc_macd_hist = 0.0
                btc_prev_macd_hist = None
                
                if btc_timestamps and len(btc_timestamps) > 0:
                    # Find closest BTC timestamp
                    current_ts = timestamps[i]
                    closest_idx = 0
                    min_diff = float('inf')
                    for idx, btc_ts in enumerate(btc_timestamps):
                        diff = abs((datetime.fromisoformat(current_ts.replace("Z", "+00:00")) - 
                                   datetime.fromisoformat(btc_ts.replace("Z", "+00:00"))).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            closest_idx = idx
                    
                    if closest_idx < len(btc_rsi_values):
                        btc_rsi_h4 = btc_rsi_values[closest_idx]
                    if closest_idx < len(btc_macd_hist_values):
                        btc_macd_hist = btc_macd_hist_values[closest_idx]
                    if closest_idx > 0 and closest_idx - 1 < len(btc_macd_hist_values):
                        btc_prev_macd_hist = btc_macd_hist_values[closest_idx - 1]
                
                # Quyết định action (dùng logic _eth_decide_action)
                zones = (sell_low, sell_high, buy_low, buy_high, recent_low, recent_high)
                decision = _eth_decide_action(
                    price=price,
                    rsi_h4=rsi,
                    macd_hist=macd_hist,
                    prev_macd_hist=prev_macd_hist,
                    zones=zones,
                    btc_rsi_h4=btc_rsi_h4,
                    btc_macd_hist=btc_macd_hist,
                    btc_prev_macd_hist=btc_prev_macd_hist,
                )
                
                action = decision["action"]
                
                if action == "BUY":
                    buy_count += 1
                    signals.append({
                        "time": timestamps[i],
                        "price": price,
                        "action": "BUY",
                        "rsi": rsi,
                        "macd_hist": macd_hist,
                    })
                elif action == "SELL":
                    sell_count += 1
                    signals.append({
                        "time": timestamps[i],
                        "price": price,
                        "action": "SELL",
                        "rsi": rsi,
                        "macd_hist": macd_hist,
                    })
            except Exception as e:
                logging.warning(f"[BACKTEST] Error processing candle {i}: {e}")
                continue
        
        return JSONResponse({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "total_candles": len(closes),
            "signals": signals,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "prices": closes,
            "timestamps": timestamps,
        })
        
    except Exception as e:
        logging.error(f"[BACKTEST] Error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# ------------------------------------------------------------------
# 5) SCHEDULER INIT
# ------------------------------------------------------------------
scheduler = BackgroundScheduler()
init_inline_rsi_dual(app, scheduler)

# ------------------------------------------------------------------
# 6) STARTUP/SHUTDOWN EVENTS
# ------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize database and start scheduler on startup"""
    try:
        await init_db()
        logging.info("[STARTUP] Database initialized successfully")
    except Exception as e:
        logging.error(f"[STARTUP] Error initializing database: {e}")
    
    # Start scheduler
    if not scheduler.running:
        scheduler.start()
        logging.info("[STARTUP] Scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database pool and stop scheduler on shutdown"""
    global _db_pool
    
    # Shutdown scheduler
    if scheduler.running:
        scheduler.shutdown()
        logging.info("[SHUTDOWN] Scheduler stopped")
    
    # Close database pool
    if _db_pool:
        await _db_pool.close()
        logging.info("[SHUTDOWN] Database pool closed")

# ------------------------------------------------------------------
# 7) ENTRY POINT (for local testing only)
# ------------------------------------------------------------------
# Note: On Koyeb, the app is run via: uvicorn crypto:app --host 0.0.0.0 --port $PORT
# This block is only executed when running: python crypto.py (local development)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10001)),
        reload=False,  # Disable reload for production-like testing
        workers=1,
    )


