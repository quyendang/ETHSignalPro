import os
import math
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal, Dict, Any, List, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from sqlalchemy import (
    String, Float, Integer, DateTime, JSON, select, UniqueConstraint
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker


# -----------------------------
# Settings
# -----------------------------
class Settings(BaseSettings):
    PUSHOVER_TOKEN: str = os.environ.get("PUSHOVER_TOKEN", "")
    PUSHOVER_USER: str = os.environ.get("PUSHOVER_USER", "")
    PUSHOVER_DEVICE: str = os.environ.get("PUSHOVER_DEVICE", "")
    PUSHOVER_SOUND: str = os.environ.get("PUSHOVER_SOUND", "cash")

    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgres://koyeb-adm:*******@ep-aged-bush-a1op42oy.ap-southeast-1.pg.koyeb.app/koyebdb"
    )

    POLL_SECONDS: int = int(os.environ.get("POLL_SECONDS", "60"))
    SIGNAL_COOLDOWN_MINUTES: int = int(os.environ.get("SIGNAL_COOLDOWN_MINUTES", "10"))
    MIN_PRICE_MOVE_PCT: float = float(os.environ.get("MIN_PRICE_MOVE_PCT", "0.15"))  # %

    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # Telegram notify (optional)
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Strategy thresholds
    ETH_RSI_BUY_SPOT: float = 32.0
    ETH_RSI_BORROW_MIN: float = 35.0
    BTC_RSI_PANIC: float = 25.0

    EMA_FAST: int = 34
    EMA_SLOW: int = 200

    BACKTEST_LOOKAHEAD_BARS: int = 6  # 6*4H = 24h

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("safe-borrow-bot")


# -----------------------------
# DB setup
# -----------------------------
Base = declarative_base()


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(_to_asyncpg_url(settings.DATABASE_URL), pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    eth_price: Mapped[float] = mapped_column(Float)
    btc_price: Mapped[float] = mapped_column(Float)
    ethbtc_price: Mapped[float] = mapped_column(Float)

    eth_rsi_4h: Mapped[float] = mapped_column(Float)
    btc_rsi_4h: Mapped[float] = mapped_column(Float)
    ethbtc_rsi_4h: Mapped[float] = mapped_column(Float)

    eth_macd_hist_4h: Mapped[float] = mapped_column(Float)
    btc_macd_hist_4h: Mapped[float] = mapped_column(Float)
    ethbtc_macd_hist_4h: Mapped[float] = mapped_column(Float)

    eth_bb_lower_4h: Mapped[float] = mapped_column(Float)
    eth_bb_mid_4h: Mapped[float] = mapped_column(Float)
    eth_bb_upper_4h: Mapped[float] = mapped_column(Float)

    btc_ema34_4h: Mapped[float] = mapped_column(Float)
    btc_ema200_4h: Mapped[float] = mapped_column(Float)
    btc_breakdown: Mapped[bool] = mapped_column(Integer)  # 0/1


class SignalLog(Base):
    __tablename__ = "signal_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    action: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(String(500))
    meta: Mapped[dict] = mapped_column(JSON)

    eth_price: Mapped[float] = mapped_column(Float)
    btc_price: Mapped[float] = mapped_column(Float)
    ethbtc_price: Mapped[float] = mapped_column(Float)


class AlertState(Base):
    __tablename__ = "alert_state"
    __table_args__ = (UniqueConstraint("key", name="uq_alert_state_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    last_sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_sent_price: Mapped[float] = mapped_column(Float)


# -----------------------------
# Templates (dashboard)
# -----------------------------
templates = Jinja2Templates(directory="templates")

DASHBOARD_HTML = """\
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Safe Borrow Bot Dashboard</title>
  <style>
    body{font-family:ui-sans-serif,system-ui; margin:20px; background:#0b0f14; color:#e6edf3;}
    .row{display:flex; gap:14px; flex-wrap:wrap;}
    .card{background:#111824; border:1px solid #1f2a3a; border-radius:12px; padding:14px; min-width:280px;}
    .muted{color:#9fb1c3}
    table{width:100%; border-collapse:collapse;}
    th,td{padding:10px; border-bottom:1px solid #1f2a3a; font-size:14px;}
    .badge{padding:4px 8px; border-radius:999px; border:1px solid #2a3a52; display:inline-block}
    .buy{background:rgba(46,204,113,.12); border-color:rgba(46,204,113,.35)}
    .risk{background:rgba(231,76,60,.12); border-color:rgba(231,76,60,.35)}
    .hold{background:rgba(149,165,166,.10); border-color:rgba(149,165,166,.25)}
    a{color:#7cc2ff}
  </style>
</head>
<body>
  <h2>Safe Borrow Bot Dashboard</h2>
  <div class="muted">Auto refresh mỗi 30s. <a href="/api/signals?limit=50">API signals</a> | <a href="/api/market/latest">API market</a></div>
  <script>
    setTimeout(()=>location.reload(), 30000);
  </script>

  <div class="row" style="margin-top:14px">
    <div class="card">
      <div class="muted">Latest Market</div>
      <div style="font-size:22px; margin-top:6px;">ETH: {{market.eth_price}} | BTC: {{market.btc_price}}</div>
      <div class="muted" style="margin-top:6px;">ETHBTC: {{market.ethbtc_price}}</div>
      <div style="margin-top:10px;">
        <div>ETH RSI4H: <b>{{market.eth_rsi_4h}}</b> | BTC RSI4H: <b>{{market.btc_rsi_4h}}</b></div>
        <div class="muted">BTC breakdown: <b>{{market.btc_breakdown}}</b></div>
      </div>
    </div>

    <div class="card">
      <div class="muted">Latest Signal</div>
      <div style="margin-top:8px;">
        <span class="badge {{sig_class}}">{{signal.action}} ({{signal.confidence}})</span>
      </div>
      <div style="margin-top:10px;">{{signal.message}}</div>
      <div class="muted" style="margin-top:10px;">{{signal.time}}</div>
    </div>

    <div class="card">
      <div class="muted">Position</div>
      <div style="margin-top:8px;">spot_eth: <b>{{pos.spot_eth}}</b></div>
      <div>loan_usdt: <b>{{pos.loan_usdt}}</b></div>
      <div>avg_entry: <b>{{pos.avg_entry}}</b></div>
      <div class="muted" style="margin-top:8px;">Update via POST /position</div>
    </div>
  </div>

  <div class="card" style="margin-top:14px">
    <div class="muted">Recent signals</div>
    <table style="margin-top:10px">
      <thead>
        <tr>
          <th>Time</th><th>Action</th><th>ETH</th><th>BTC</th><th>Message</th>
        </tr>
      </thead>
      <tbody>
        {% for s in signals %}
        <tr>
          <td class="muted">{{s.time}}</td>
          <td><span class="badge {% if 'BORROW' in s.action or 'BUY' in s.action %}buy{% elif s.action=='REDUCE_RISK' %}risk{% else %}hold{% endif %}">{{s.action}}</span></td>
          <td>{{s.eth_price}}</td>
          <td>{{s.btc_price}}</td>
          <td class="muted">{{s.message}}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

</body>
</html>
"""


# -----------------------------
# Binance helpers
# -----------------------------
BINANCE = "https://api.binance.com"


async def fetch_klines(symbol: str, interval: str, limit: int = 300) -> np.ndarray:
    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    closes = np.array([float(x[4]) for x in data], dtype=np.float64)
    highs = np.array([float(x[2]) for x in data], dtype=np.float64)
    lows = np.array([float(x[3]) for x in data], dtype=np.float64)
    return np.vstack([closes, highs, lows])


# -----------------------------
# Indicators
# -----------------------------
def ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) < period + 2:
        return np.full_like(series, np.nan)
    alpha = 2 / (period + 1)
    out = np.empty_like(series, dtype=np.float64)
    out[:] = np.nan
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    if len(series) < period + 2:
        return np.full_like(series, np.nan)
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.empty_like(series, dtype=np.float64)
    avg_loss = np.empty_like(series, dtype=np.float64)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])

    for i in range(period + 1, len(series)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out


def macd_hist(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    ef = ema(series, fast)
    es = ema(series, slow)
    macd_line = ef - es
    sig = ema(macd_line[~np.isnan(macd_line)], signal)
    sig_aligned = np.full_like(macd_line, np.nan)
    sig_aligned[-len(sig):] = sig
    hist = macd_line - sig_aligned
    return hist


def bollinger(series: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(series) < period:
        nan = np.full_like(series, np.nan)
        return nan, nan, nan
    mid = np.full_like(series, np.nan)
    upper = np.full_like(series, np.nan)
    lower = np.full_like(series, np.nan)
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        m = window.mean()
        s = window.std(ddof=0)
        mid[i] = m
        upper[i] = m + mult * s
        lower[i] = m - mult * s
    return lower, mid, upper


def is_macd_hist_rising(hist: np.ndarray, bars: int = 3) -> bool:
    h = hist[~np.isnan(hist)]
    if len(h) < bars + 1:
        return False
    last = h[-(bars+1):]
    return all(last[i] < last[i + 1] for i in range(len(last) - 1))


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


# -----------------------------
# Strategy
# -----------------------------
SignalAction = Literal["HOLD", "BUY_SPOT", "BUYBACK", "BORROW_BUY", "REDUCE_RISK"]


@dataclass
class Market:
    time: datetime
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    btc_rsi_4h: float
    ethbtc_rsi_4h: float

    eth_macd_hist_4h: float
    btc_macd_hist_4h: float
    ethbtc_macd_hist_4h: float

    eth_bb_lower_4h: float
    eth_bb_mid_4h: float
    eth_bb_upper_4h: float

    btc_ema34_4h: float
    btc_ema200_4h: float
    btc_breakdown: bool

    eth_macd_hist_rising: bool
    btc_rebound: bool
    ethbtc_no_lower_low: bool


class Signal(BaseModel):
    action: SignalAction
    confidence: Literal["low", "med", "high"]
    message: str
    meta: Dict[str, Any]


def detect_higher_low(closes: np.ndarray, lookback: int = 30) -> bool:
    x = closes[-lookback:]
    if len(x) < 10:
        return False
    mins = []
    for i in range(2, len(x) - 2):
        if x[i] < x[i-1] and x[i] < x[i+1] and x[i] < x[i-2] and x[i] < x[i+2]:
            mins.append((i, x[i]))
    if len(mins) < 2:
        return False
    (_, low1), (_, low2) = mins[-2], mins[-1]
    return low2 > low1


def detect_no_lower_low(closes: np.ndarray, lookback: int = 40) -> bool:
    x = closes[-lookback:]
    if len(x) < 10:
        return True
    lows = []
    for i in range(2, len(x) - 2):
        if x[i] < x[i-1] and x[i] < x[i+1]:
            lows.append(x[i])
    if len(lows) < 2:
        return True
    return lows[-1] >= lows[-2]


def compute_market(eth_4h: np.ndarray, btc_4h: np.ndarray, ethbtc_4h: np.ndarray) -> Market:
    now = datetime.now(timezone.utc)

    eth_close = eth_4h[0]
    btc_close = btc_4h[0]
    ethbtc_close = ethbtc_4h[0]

    eth_price = float(eth_close[-1])
    btc_price = float(btc_close[-1])
    ethbtc_price = float(ethbtc_close[-1])

    eth_r = rsi(eth_close, 14)
    btc_r = rsi(btc_close, 14)
    ethbtc_r = rsi(ethbtc_close, 14)

    eth_hist = macd_hist(eth_close)
    btc_hist = macd_hist(btc_close)
    ethbtc_hist = macd_hist(ethbtc_close)

    bb_l, bb_m, bb_u = bollinger(eth_close, 20, 2)

    btc_ema34 = ema(btc_close, settings.EMA_FAST)
    btc_ema200 = ema(btc_close, settings.EMA_SLOW)
    btc_ema34_last = float(btc_ema34[-1]) if not math.isnan(btc_ema34[-1]) else float("nan")
    btc_ema200_last = float(btc_ema200[-1]) if not math.isnan(btc_ema200[-1]) else float("nan")

    btc_breakdown = bool((btc_price < btc_ema34_last) and (btc_ema34_last < btc_ema200_last))
    btc_rebound = bool((btc_price > btc_ema34_last) or detect_higher_low(btc_close))

    eth_hist_rising = is_macd_hist_rising(eth_hist, bars=3)
    ethbtc_ok = detect_no_lower_low(ethbtc_close)

    return Market(
        time=now,
        eth_price=eth_price,
        btc_price=btc_price,
        ethbtc_price=ethbtc_price,

        eth_rsi_4h=float(eth_r[-1]),
        btc_rsi_4h=float(btc_r[-1]),
        ethbtc_rsi_4h=float(ethbtc_r[-1]),

        eth_macd_hist_4h=float(eth_hist[-1]),
        btc_macd_hist_4h=float(btc_hist[-1]),
        ethbtc_macd_hist_4h=float(ethbtc_hist[-1]),

        eth_bb_lower_4h=float(bb_l[-1]),
        eth_bb_mid_4h=float(bb_m[-1]),
        eth_bb_upper_4h=float(bb_u[-1]),

        btc_ema34_4h=btc_ema34_last,
        btc_ema200_4h=btc_ema200_last,
        btc_breakdown=btc_breakdown,

        eth_macd_hist_rising=eth_hist_rising,
        btc_rebound=btc_rebound,
        ethbtc_no_lower_low=ethbtc_ok
    )


def decide_signal(m: Market, position: Dict[str, Any]) -> Signal:
    spot_eth = float(position.get("spot_eth", 0) or 0)
    loan_usdt = float(position.get("loan_usdt", 0) or 0)
    avg_entry = float(position.get("avg_entry", 0) or 0)

    if spot_eth > 0:
        collateral = spot_eth * m.eth_price
        ltv = (loan_usdt / collateral * 100.0) if collateral > 0 else 0.0
    else:
        ltv = float(position.get("ltv_pct", 0) or 0)

    in_buy_zone = (m.eth_price <= m.eth_bb_lower_4h * 1.01) or (m.eth_rsi_4h <= settings.ETH_RSI_BUY_SPOT)
    btc_panic = m.btc_rsi_4h <= settings.BTC_RSI_PANIC
    macro_risk = m.btc_breakdown or btc_panic

    # BORROW_BUY (strict)
    borrow_allowed = (ltv < 15.0)
    eth_stable = (m.eth_rsi_4h >= settings.ETH_RSI_BORROW_MIN) and m.eth_macd_hist_rising
    borrow_conditions = borrow_allowed and m.btc_rebound and (not m.btc_breakdown) and eth_stable and m.ethbtc_no_lower_low

    if borrow_conditions and in_buy_zone:
        return Signal(
            action="BORROW_BUY",
            confidence="high" if (not macro_risk) else "med",
            message="BORROW_BUY (an toàn): ETH vùng mua + BTC rebound + ETH ổn định (RSI đủ & MACD hist tăng).",
            meta={
                "ltv_pct": round(ltv, 2),
                "in_buy_zone": in_buy_zone,
                "btc_rebound": m.btc_rebound,
                "btc_breakdown": m.btc_breakdown,
                "eth_stable": eth_stable,
                "eth_macd_hist_rising": m.eth_macd_hist_rising,
                "ethbtc_no_lower_low": m.ethbtc_no_lower_low,
                "eth_bb_lower_4h": round(m.eth_bb_lower_4h, 2),
                "eth_rsi_4h": round(m.eth_rsi_4h, 2),
                "btc_rsi_4h": round(m.btc_rsi_4h, 2),
                "btc_ema34_4h": round(m.btc_ema34_4h, 2) if not math.isnan(m.btc_ema34_4h) else None,
                "btc_ema200_4h": round(m.btc_ema200_4h, 2) if not math.isnan(m.btc_ema200_4h) else None,
            }
        )

    # BUYBACK / BUY_SPOT
    if spot_eth > 0 and in_buy_zone and (not m.btc_breakdown):
        return Signal(
            action="BUYBACK",
            confidence="med" if macro_risk else "high",
            message="BUYBACK: ETH về vùng mua lại, BTC chưa breakdown. Ưu tiên DCA nhẹ / mua lại phần đã bán.",
            meta={
                "ltv_pct": round(ltv, 2),
                "in_buy_zone": in_buy_zone,
                "eth_bb_lower_4h": round(m.eth_bb_lower_4h, 2),
                "eth_rsi_4h": round(m.eth_rsi_4h, 2),
                "btc_rsi_4h": round(m.btc_rsi_4h, 2),
                "btc_breakdown": m.btc_breakdown,
            }
        )

    if spot_eth <= 0 and in_buy_zone and (not m.btc_breakdown):
        return Signal(
            action="BUY_SPOT",
            confidence="med" if macro_risk else "high",
            message="BUY_SPOT: ETH vào vùng mua (không vay).",
            meta={
                "in_buy_zone": in_buy_zone,
                "eth_bb_lower_4h": round(m.eth_bb_lower_4h, 2),
                "eth_rsi_4h": round(m.eth_rsi_4h, 2),
                "btc_rsi_4h": round(m.btc_rsi_4h, 2),
                "btc_breakdown": m.btc_breakdown,
            }
        )

    if loan_usdt > 0 and m.btc_breakdown:
        return Signal(
            action="REDUCE_RISK",
            confidence="high",
            message="REDUCE_RISK: BTC breakdown trong khi đang vay → ưu tiên giảm rủi ro/giảm nợ, không tăng vị thế.",
            meta={
                "ltv_pct": round(ltv, 2),
                "btc_breakdown": True,
                "btc_rsi_4h": round(m.btc_rsi_4h, 2),
                "eth_rsi_4h": round(m.eth_rsi_4h, 2),
            }
        )

    return Signal(
        action="HOLD",
        confidence="low",
        message="HOLD: Chưa có điều kiện rõ ràng cho vay/mua an toàn.",
        meta={
            "in_buy_zone": in_buy_zone,
            "btc_rebound": m.btc_rebound,
            "btc_breakdown": m.btc_breakdown,
            "eth_macd_hist_rising": m.eth_macd_hist_rising,
            "ethbtc_no_lower_low": m.ethbtc_no_lower_low,
        }
    )


# -----------------------------
# Telegram Notify
# -----------------------------
async def send_pushover(title: str, message: str):
    if not settings.PUSHOVER_TOKEN or not settings.PUSHOVER_USER:
        return
    data = {
        "token": settings.PUSHOVER_TOKEN,
        "user": settings.PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": 0,
        "sound": settings.PUSHOVER_SOUND or "cash",
    }
    if settings.PUSHOVER_DEVICE:
        data["device"] = settings.PUSHOVER_DEVICE

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post("https://api.pushover.net/1/messages.json", data=data)
    except Exception as e:
        log.warning("Pushover notify failed: %s", e)


def format_pushover(sig: "Signal", m: "Market") -> tuple[str, str]:
    title = f"{sig.action} ({sig.confidence}) | ETH {m.eth_price:.0f}"
    msg = (
        f"{sig.message}\n\n"
        f"ETH: {m.eth_price:.2f}\n"
        f"BTC: {m.btc_price:.2f}\n"
        f"ETHBTC: {m.ethbtc_price:.6f}\n"
        f"ETH RSI4H: {m.eth_rsi_4h:.2f} | BTC RSI4H: {m.btc_rsi_4h:.2f}\n"
        f"BTC breakdown: {m.btc_breakdown} | BTC rebound: {m.btc_rebound}\n"
        f"Time: {m.time.isoformat()}"
    )
    return title, msg



# -----------------------------
# Anti-spam / cooldown
# -----------------------------
async def should_emit(session: AsyncSession, action: str, price: float) -> bool:
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=settings.SIGNAL_COOLDOWN_MINUTES)

    q = select(AlertState).where(AlertState.key == action)
    row = (await session.execute(q)).scalars().first()

    if not row:
        return True

    time_since = now - row.last_sent_at
    if time_since >= cooldown:
        return True

    move = abs(pct_change(price, row.last_sent_price))
    if time_since >= timedelta(minutes=2) and move >= settings.MIN_PRICE_MOVE_PCT:
        return True

    return False


async def update_emit_state(session: AsyncSession, action: str, price: float):
    now = datetime.now(timezone.utc)
    q = select(AlertState).where(AlertState.key == action)
    row = (await session.execute(q)).scalars().first()
    if not row:
        row = AlertState(key=action, last_sent_at=now, last_sent_price=price)
        session.add(row)
    else:
        row.last_sent_at = now
        row.last_sent_price = price


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Safe Borrow Bot (Koyeb Postgres + Telegram + Dashboard)")


POSITION_STATE = {
    "spot_eth": 100,
    "loan_usdt": 30000,
    "avg_entry": 3150
}


class PositionUpdate(BaseModel):
    spot_eth: float = 0
    loan_usdt: float = 0
    avg_entry: float = 0


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ensure templates dir exists even on Koyeb ephemeral fs
    os.makedirs("templates", exist_ok=True)
    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(DASHBOARD_HTML)

    asyncio.create_task(poller_loop())
    log.info("Bot started. Polling every %s seconds.", settings.POLL_SECONDS)


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/position")
async def get_position():
    return POSITION_STATE


@app.post("/position")
async def set_position(pos: PositionUpdate):
    POSITION_STATE["spot_eth"] = pos.spot_eth
    POSITION_STATE["loan_usdt"] = pos.loan_usdt
    POSITION_STATE["avg_entry"] = pos.avg_entry
    return {"ok": True, "position": POSITION_STATE}


@app.get("/api/market/latest")
async def api_latest_market():
    async with AsyncSessionLocal() as session:
        q = select(MarketSnapshot).order_by(MarketSnapshot.time.desc()).limit(1)
        row = (await session.execute(q)).scalars().first()
        if not row:
            return {"ok": True, "market": None}
        return {
            "ok": True,
            "time": row.time.isoformat(),
            "market": {
                "eth_price": row.eth_price,
                "btc_price": row.btc_price,
                "ethbtc_price": row.ethbtc_price,
                "eth_rsi_4h": row.eth_rsi_4h,
                "btc_rsi_4h": row.btc_rsi_4h,
                "ethbtc_rsi_4h": row.ethbtc_rsi_4h,
                "eth_bb_lower_4h": row.eth_bb_lower_4h,
                "eth_bb_mid_4h": row.eth_bb_mid_4h,
                "eth_bb_upper_4h": row.eth_bb_upper_4h,
                "btc_ema34_4h": row.btc_ema34_4h,
                "btc_ema200_4h": row.btc_ema200_4h,
                "btc_breakdown": bool(row.btc_breakdown),
            }
        }


@app.get("/api/signals")
async def api_signals(limit: int = Query(30, ge=1, le=200)):
    async with AsyncSessionLocal() as session:
        q = select(SignalLog).order_by(SignalLog.time.desc()).limit(limit)
        rows = (await session.execute(q)).scalars().all()
        return {
            "ok": True,
            "signals": [
                {
                    "time": r.time.isoformat(),
                    "action": r.action,
                    "confidence": r.confidence,
                    "message": r.message,
                    "meta": r.meta,
                    "eth_price": r.eth_price,
                    "btc_price": r.btc_price,
                    "ethbtc_price": r.ethbtc_price,
                }
                for r in rows
            ]
        }


@app.get("/api/signal/latest")
async def api_latest_signal():
    async with AsyncSessionLocal() as session:
        q = select(SignalLog).order_by(SignalLog.time.desc()).limit(1)
        row = (await session.execute(q)).scalars().first()
        if not row:
            return {"ok": True, "signal": None}
        return {
            "ok": True,
            "time": row.time.isoformat(),
            "signal": {
                "action": row.action,
                "confidence": row.confidence,
                "message": row.message,
                "meta": row.meta
            },
            "market": {
                "eth_price": row.eth_price,
                "btc_price": row.btc_price,
                "ethbtc_price": row.ethbtc_price
            }
        }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    async with AsyncSessionLocal() as session:
        q1 = select(MarketSnapshot).order_by(MarketSnapshot.time.desc()).limit(1)
        mrow = (await session.execute(q1)).scalars().first()

        q2 = select(SignalLog).order_by(SignalLog.time.desc()).limit(1)
        srow = (await session.execute(q2)).scalars().first()

        q3 = select(SignalLog).order_by(SignalLog.time.desc()).limit(20)
        recent = (await session.execute(q3)).scalars().all()

    market = {
        "eth_price": round(mrow.eth_price, 2) if mrow else None,
        "btc_price": round(mrow.btc_price, 2) if mrow else None,
        "ethbtc_price": round(mrow.ethbtc_price, 6) if mrow else None,
        "eth_rsi_4h": round(mrow.eth_rsi_4h, 2) if mrow else None,
        "btc_rsi_4h": round(mrow.btc_rsi_4h, 2) if mrow else None,
        "btc_breakdown": bool(mrow.btc_breakdown) if mrow else None,
    }

    signal = {
        "time": srow.time.isoformat() if srow else None,
        "action": srow.action if srow else "HOLD",
        "confidence": srow.confidence if srow else "low",
        "message": srow.message if srow else "Chưa có signal.",
    }

    sig_class = "hold"
    if "BORROW" in signal["action"] or "BUY" in signal["action"]:
        sig_class = "buy"
    if signal["action"] == "REDUCE_RISK":
        sig_class = "risk"

    signals = [
        {
            "time": r.time.isoformat(),
            "action": r.action,
            "message": r.message,
            "eth_price": round(r.eth_price, 2),
            "btc_price": round(r.btc_price, 2),
        }
        for r in recent
    ]

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "market": market,
            "signal": signal,
            "signals": signals,
            "pos": POSITION_STATE,
            "sig_class": sig_class
        }
    )


# -----------------------------
# Poller loop
# -----------------------------
async def poll_once():
    eth_4h = await fetch_klines("ETHUSDT", "4h", limit=300)
    btc_4h = await fetch_klines("BTCUSDT", "4h", limit=300)
    ethbtc_4h = await fetch_klines("ETHBTC", "4h", limit=300)

    m = compute_market(eth_4h, btc_4h, ethbtc_4h)
    sig = decide_signal(m, POSITION_STATE)

    async with AsyncSessionLocal() as session:
        snap = MarketSnapshot(
            time=m.time,
            eth_price=m.eth_price,
            btc_price=m.btc_price,
            ethbtc_price=m.ethbtc_price,

            eth_rsi_4h=m.eth_rsi_4h,
            btc_rsi_4h=m.btc_rsi_4h,
            ethbtc_rsi_4h=m.ethbtc_rsi_4h,

            eth_macd_hist_4h=m.eth_macd_hist_4h,
            btc_macd_hist_4h=m.btc_macd_hist_4h,
            ethbtc_macd_hist_4h=m.ethbtc_macd_hist_4h,

            eth_bb_lower_4h=m.eth_bb_lower_4h,
            eth_bb_mid_4h=m.eth_bb_mid_4h,
            eth_bb_upper_4h=m.eth_bb_upper_4h,

            btc_ema34_4h=m.btc_ema34_4h if not math.isnan(m.btc_ema34_4h) else 0.0,
            btc_ema200_4h=m.btc_ema200_4h if not math.isnan(m.btc_ema200_4h) else 0.0,
            btc_breakdown=1 if m.btc_breakdown else 0,
        )
        session.add(snap)

        if sig.action != "HOLD":
            ok = await should_emit(session, sig.action, m.eth_price)
            if ok:
                session.add(SignalLog(
                    time=m.time,
                    action=sig.action,
                    confidence=sig.confidence,
                    message=sig.message,
                    meta=sig.meta,
                    eth_price=m.eth_price,
                    btc_price=m.btc_price,
                    ethbtc_price=m.ethbtc_price
                ))
                await update_emit_state(session, sig.action, m.eth_price)

                log.info("EMIT %s | ETH %.2f | BTC %.2f", sig.action, m.eth_price, m.btc_price)
                title, msg = format_pushover(sig, m)
                await send_pushover(title, msg)
            else:
                log.info("SKIP (cooldown) %s | ETH %.2f", sig.action, m.eth_price)

        await session.commit()


async def poller_loop():
    while True:
        try:
            await poll_once()
        except Exception as e:
            log.exception("Poll error: %s", e)
        await asyncio.sleep(settings.POLL_SECONDS)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
