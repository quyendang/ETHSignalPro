import os
import math
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import httpx
import psycopg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

# =============================
# Config
# =============================
APP_TITLE = os.getenv("APP_TITLE", "ETHSignalPro")
PORT = int(os.getenv("PORT", "8000"))
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "60"))

# Notifications
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "").strip()
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "").strip()
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Koyeb Postgres
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Position (for personalization)
SPOT_ETH = float(os.getenv("SPOT_ETH", "100") or 0)
AVG_ENTRY = float(os.getenv("AVG_ENTRY", "3150") or 0)

# Loan sizing limits (suggestion only)
MAX_LOAN_USDT = float(os.getenv("MAX_LOAN_USDT", "0") or 0)
MAX_LOAN_PCT_NAV = float(os.getenv("MAX_LOAN_PCT_NAV", "0.25") or 0.25)

# Cooldown
SELL_COOLDOWN_SECONDS = int(os.getenv("SELL_COOLDOWN_SECONDS", "14400"))  # 4h

# =============================
# Utilities
# =============================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sanitize_for_json(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, dict):
        return {k: sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize_for_json(v) for v in x]
    return x

# =============================
# Indicators
# =============================
def ema(series: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def rsi(series: np.ndarray, period: int = 14) -> float:
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    v = out.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")

def macd_hist(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    s = pd.Series(series)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])

def bollinger(series: np.ndarray, period: int = 20, k: float = 2.0) -> Tuple[float, float, float]:
    s = pd.Series(series)
    mid = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std(ddof=0).iloc[-1]
    if pd.isna(mid) or pd.isna(std):
        return float("nan"), float("nan"), float("nan")
    lower = float(mid - k * std)
    upper = float(mid + k * std)
    return float(lower), float(mid), float(upper)

# =============================
# Bybit Public Client (no API key)
# =============================
class BybitPublic:
    BASE = "https://api.bybit.com"
    INTERVAL_MAP = {
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }

    def __init__(self):
        self._client = httpx.Client(timeout=20.0)

    def close(self):
        self._client.close()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.BASE + path
        r = self._client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> List[Dict[str, Any]]:
        iv = self.INTERVAL_MAP.get(interval)
        if not iv:
            raise ValueError(f"Unsupported interval: {interval}")

        data = self._get(
            "/v5/market/kline",
            {
                "category": "spot",
                "symbol": symbol,
                "interval": iv,
                "limit": limit,
            },
        )

        rows = data.get("result", {}).get("list", [])
        candles = []
        for row in rows:
            ts_ms = int(row[0])
            candles.append(
                {
                    "timestamp": ts_ms,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )

        candles.sort(key=lambda x: x["timestamp"])
        return candles

# =============================
# DB (Koyeb Postgres)
# =============================
def db_init():
    if not DATABASE_URL:
        print("[DB] DATABASE_URL not set -> DB logging disabled")
        return
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    action TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    message TEXT NOT NULL,
                    meta JSONB,
                    market JSONB,
                    position JSONB
                );
                """)
                conn.commit()
        print("[DB] init ok")
    except Exception as e:
        print(f"[DB] init error: {e}")

def db_log_signal(payload: Dict[str, Any]):
    if not DATABASE_URL:
        return
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO signals (created_at, action, confidence, message, meta, market, position)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb);
                    """,
                    (
                        payload.get("created_at"),
                        payload.get("action"),
                        payload.get("confidence"),
                        payload.get("message"),
                        json.dumps(payload.get("meta", {}), ensure_ascii=False),
                        json.dumps(payload.get("market", {}), ensure_ascii=False),
                        json.dumps(payload.get("position", {}), ensure_ascii=False),
                    ),
                )
                conn.commit()
    except Exception as e:
        print(f"[DB] insert error: {e}")

# =============================
# Notifications
# =============================
def pushover_notify(title: str, message: str):
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
        httpx.post("https://api.pushover.net/1/messages.json", data=data, timeout=15.0)
    except Exception:
        pass

def telegram_notify(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        httpx.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15.0)
    except Exception:
        pass

# =============================
# State + Strategy
# =============================
@dataclass
class MarketSnapshot:
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    eth_rsi_1d: float
    btc_rsi_4h: float
    btc_rsi_1d: float
    ethbtc_rsi_4h: float
    ethbtc_rsi_1d: float

    eth_macd_hist_4h: float
    btc_macd_hist_4h: float
    ethbtc_macd_hist_4h: float

    eth_bb_lower_4h: float
    eth_bb_mid_4h: float
    eth_bb_upper_4h: float
    eth_ma200_1d: float

@dataclass
class Signal:
    action: str          # HOLD / SELL_ROTATE / BUYBACK / LOAN_VALUE / LOAN_MOMENTUM / REPAY
    confidence: str      # low/med/high
    message: str
    meta: Dict[str, Any]

class StrategyEngine:
    def __init__(self):
        self.last_sell_ts: Optional[int] = None

    def _in_cooldown(self, now_ts: int) -> bool:
        if self.last_sell_ts is None:
            return False
        return (now_ts - self.last_sell_ts) < SELL_COOLDOWN_SECONDS

    def decide(self, m: MarketSnapshot, position: Dict[str, Any]) -> Signal:
        spot_eth = float(position.get("spot_eth", SPOT_ETH))
        loan_usdt = float(position.get("loan_usdt", 0.0))
        avg_entry = float(position.get("avg_entry", AVG_ENTRY))

        now_ts = int(time.time())

        # Dynamic anchors
        bb_sell_zone = (m.eth_bb_upper_4h * 0.995, m.eth_bb_upper_4h * 1.02)
        bb_buy_zone = (m.eth_bb_lower_4h, m.eth_bb_mid_4h)

        # Extra “hard” zones to keep stable across months
        hard_sell1 = (3350, 3450)
        hard_sell2 = (3550, 3700)
        hard_buy1 = (3000, 3050)
        hard_buy2 = (2850, 2920)

        # 1) REPAY (if already loaned)
        if loan_usdt > 0:
            if (m.eth_price > m.eth_bb_mid_4h) and (m.eth_rsi_4h >= 55):
                return Signal(
                    action="REPAY",
                    confidence="med",
                    message=f"ETH hồi đủ để trả nợ: giá {m.eth_price:.0f} > BB_mid {m.eth_bb_mid_4h:.0f}, RSI4H {m.eth_rsi_4h:.1f}. Ưu tiên chốt 1 phần giảm áp lực.",
                    meta={
                        "suggested_sell_eth": round(min(max(spot_eth * 0.08, 1.0), 10.0), 3),
                        "loan_usdt": loan_usdt,
                        "eth_price": m.eth_price,
                    },
                )

        # 2) SELL_ROTATE (cooldown 4h)
        if not self._in_cooldown(now_ts):
            eth_at_resistance = (
                (hard_sell1[0] <= m.eth_price <= hard_sell1[1]) or
                (hard_sell2[0] <= m.eth_price <= hard_sell2[1]) or
                (bb_sell_zone[0] <= m.eth_price <= bb_sell_zone[1])
            )
            ethbtc_not_strong = (m.ethbtc_rsi_4h <= 55) or (m.ethbtc_macd_hist_4h < 0)

            if eth_at_resistance and ethbtc_not_strong and spot_eth > 0:
                if spot_eth >= 120:
                    sell_eth = 30
                elif spot_eth >= 80:
                    sell_eth = 20
                else:
                    sell_eth = spot_eth * 0.2

                self.last_sell_ts = now_ts
                return Signal(
                    action="SELL_ROTATE",
                    confidence="med",
                    message=f"Vùng bán xoay vòng: ETH {m.eth_price:.0f} chạm kháng cự; ETHBTC chưa mạnh. Bán một phần chờ mua lại thấp.",
                    meta={
                        "sell_eth": round(sell_eth, 3),
                        "sell_zones": {"hard1": hard_sell1, "hard2": hard_sell2, "bb": (round(bb_sell_zone[0]), round(bb_sell_zone[1]))},
                        "ethbtc": m.ethbtc_price,
                    },
                )

        # 3) BUYBACK
        buyback_ok = (
            (hard_buy1[0] <= m.eth_price <= hard_buy1[1]) or
            (hard_buy2[0] <= m.eth_price <= hard_buy2[1]) or
            (m.eth_price <= m.eth_bb_lower_4h * 1.01)
        )
        
        # BTC safety: không buyback khi BTC đang breakdown mạnh
        btc_breakdown = (m.btc_macd_hist_4h < 0) and (m.btc_rsi_4h < 30)
        
        if buyback_ok and not btc_breakdown:
            return Signal(
                action="BUYBACK",
                confidence="med",
                message=f"Vùng mua lại: ETH {m.eth_price:.0f} vào hỗ trợ/BB lower và BTC không breakdown mạnh. Ưu tiên mua lại phần đã bán.",
                meta={
                    "buy_zones": {"hard1": hard_buy1, "hard2": hard_buy2, "bb_lower": round(m.eth_bb_lower_4h)},
                    "eth_rsi_4h": m.eth_rsi_4h,
                    "btc_price": m.btc_price,
                    "btc_breakdown": btc_breakdown,
                },
            )
        
        # Nếu ETH vào vùng mua nhưng BTC đang breakdown -> WAIT
        if buyback_ok and btc_breakdown:
            return Signal(
                action="HOLD",
                confidence="med",
                message=f"ETH vào vùng mua ({m.eth_price:.0f}) nhưng BTC đang breakdown ({m.btc_price:.0f}). Không buyback, chờ BTC ổn định.",
                meta={
                    "eth_price": m.eth_price,
                    "btc_price": m.btc_price,
                    "btc_rsi_4h": m.btc_rsi_4h,
                    "btc_macd_hist_4h": m.btc_macd_hist_4h,
                },
            )


        # 4) LOAN_VALUE (corrected logic)
        btc_floor = 88000 if m.btc_rsi_1d > 35 else 90000
        eth_value_zone = (m.eth_price <= 3000) or (m.eth_price <= m.eth_bb_lower_4h * 1.01)
        ethbtc_ok = (m.ethbtc_price >= 0.0325)  # don't require 0.0345 here
        btc_ok = (m.btc_price >= btc_floor)

        if loan_usdt <= 0 and MAX_LOAN_USDT > 0 and eth_value_zone and btc_ok and ethbtc_ok:
            nav_usdt = spot_eth * m.eth_price
            cap_by_nav = nav_usdt * MAX_LOAN_PCT_NAV
            loan_amt = max(0.0, min(MAX_LOAN_USDT, cap_by_nav))
            loan_amt *= 0.6  # conservative for value mode

            return Signal(
                action="LOAN_VALUE",
                confidence="med",
                message=f"VAY VALUE: ETH đang rẻ ({m.eth_price:.0f}), BTC không breakdown (>= {btc_floor:.0f}), ETHBTC không gãy đáy. Vay nhỏ để mua thêm.",
                meta={
                    "loan_usdt_suggested": round(loan_amt, 2),
                    "btc_floor": btc_floor,
                    "ethbtc": m.ethbtc_price,
                    "nav_usdt": round(nav_usdt, 2),
                    "max_pct_nav": MAX_LOAN_PCT_NAV,
                },
            )

        # 5) LOAN_MOMENTUM (follow-through)
        momentum_ok = (
            (m.ethbtc_price > 0.0345) and
            (m.ethbtc_rsi_4h >= 55) and
            (m.btc_price >= 90000) and
            (m.eth_price >= m.eth_bb_mid_4h)
        )
        if loan_usdt <= 0 and MAX_LOAN_USDT > 0 and momentum_ok:
            nav_usdt = spot_eth * m.eth_price
            cap_by_nav = nav_usdt * MAX_LOAN_PCT_NAV
            loan_amt = max(0.0, min(MAX_LOAN_USDT, cap_by_nav))
            loan_amt *= 0.5

            return Signal(
                action="LOAN_MOMENTUM",
                confidence="high",
                message="VAY MOMENTUM: ETHBTC breakout > 0.0345 và ETH reclaim BB mid, BTC ổn. Có thể vay nhỏ để theo sóng.",
                meta={
                    "loan_usdt_suggested": round(loan_amt, 2),
                    "ethbtc": m.ethbtc_price,
                    "eth_price": m.eth_price,
                    "btc_price": m.btc_price,
                },
            )

        return Signal(
            action="HOLD",
            confidence="low",
            message="Chưa có tín hiệu rõ. Ưu tiên giữ và chờ vùng support/resistance hoặc ETHBTC breakout.",
            meta={"eth_price": m.eth_price, "btc_price": m.btc_price, "ethbtc": m.ethbtc_price},
        )

# =============================
# Core Bot
# =============================
class ETHSignalPro:
    def __init__(self):
        self.client = BybitPublic()
        self.engine = StrategyEngine()

        self.position = {
            "spot_eth": SPOT_ETH,
            "loan_usdt": 0.0,
            "avg_entry": AVG_ENTRY,
        }

        self.cache_ohlcv: Dict[str, Any] = {}
        self.cache_market: Optional[MarketSnapshot] = None
        self.cache_signal: Optional[Dict[str, Any]] = None
        self.recent_signals: List[Dict[str, Any]] = []

    def build_pair_tf(self, symbol: str, tf: str, limit: int = 300) -> Dict[str, Any]:
        candles = self.client.get_klines(symbol, tf, limit=limit)
        closes = np.array([c["close"] for c in candles], dtype=float)
        latest = candles[-1] if candles else None

        ema34_v = float(ema(closes, 34)[-1]) if len(closes) else float("nan")
        ema89_v = float(ema(closes, 89)[-1]) if len(closes) else float("nan")
        ema200_v = float(ema(closes, 200)[-1]) if len(closes) else float("nan")
        rsi14_v = rsi(closes, 14) if len(closes) else float("nan")
        macd_v, macd_sig_v, macd_hist_v = macd_hist(closes) if len(closes) else (float("nan"), float("nan"), float("nan"))
        bb_l, bb_m, bb_u = bollinger(closes, 20, 2.0) if len(closes) else (float("nan"), float("nan"), float("nan"))

        return {
            "candles": candles,
            "latest": latest,
            "indicators": {
                "ema34": ema34_v,
                "ema89": ema89_v,
                "ema200": ema200_v,
                "rsi14": rsi14_v,
                "macd": macd_v,
                "macd_signal": macd_sig_v,
                "macd_hist": macd_hist_v,
                "bb_lower": bb_l,
                "bb_mid": bb_m,
                "bb_upper": bb_u,
            },
        }

    def refresh(self):
        pairs = {"ethusdt": "ETHUSDT", "btcusdt": "BTCUSDT", "ethbtc": "ETHBTC"}
        tfs = ["4h", "1d"]

        o: Dict[str, Any] = {}
        for k, sym in pairs.items():
            o[k] = {}
            for tf in tfs:
                o[k][tf] = self.build_pair_tf(sym, tf, limit=300)

        eth_price = float(o["ethusdt"]["4h"]["latest"]["close"])
        btc_price = float(o["btcusdt"]["4h"]["latest"]["close"])
        ethbtc_price = float(o["ethbtc"]["4h"]["latest"]["close"])

        m = MarketSnapshot(
            eth_price=eth_price,
            btc_price=btc_price,
            ethbtc_price=ethbtc_price,

            eth_rsi_4h=float(o["ethusdt"]["4h"]["indicators"]["rsi14"]),
            eth_rsi_1d=float(o["ethusdt"]["1d"]["indicators"]["rsi14"]),
            btc_rsi_4h=float(o["btcusdt"]["4h"]["indicators"]["rsi14"]),
            btc_rsi_1d=float(o["btcusdt"]["1d"]["indicators"]["rsi14"]),
            ethbtc_rsi_4h=float(o["ethbtc"]["4h"]["indicators"]["rsi14"]),
            ethbtc_rsi_1d=float(o["ethbtc"]["1d"]["indicators"]["rsi14"]),

            eth_macd_hist_4h=float(o["ethusdt"]["4h"]["indicators"]["macd_hist"]),
            btc_macd_hist_4h=float(o["btcusdt"]["4h"]["indicators"]["macd_hist"]),
            ethbtc_macd_hist_4h=float(o["ethbtc"]["4h"]["indicators"]["macd_hist"]),

            eth_bb_lower_4h=float(o["ethusdt"]["4h"]["indicators"]["bb_lower"]),
            eth_bb_mid_4h=float(o["ethusdt"]["4h"]["indicators"]["bb_mid"]),
            eth_bb_upper_4h=float(o["ethusdt"]["4h"]["indicators"]["bb_upper"]),
            eth_ma200_1d=float(o["ethusdt"]["1d"]["indicators"]["ema200"]),
        )

        sig = self.engine.decide(m, self.position)
        sig_obj = {
            "time": utc_now_iso(),
            "signal": asdict(sig),
            "market": asdict(m),
            "position": dict(self.position),
        }

        self.cache_ohlcv = o
        self.cache_market = m
        self.cache_signal = sig_obj

        # Notify + DB log only when action != HOLD
        if sig.action != "HOLD":
            title = f"[ETHSignalPro] {sig.action} ({sig.confidence})"
            msg = (
                sig.message + "\n"
                f"ETH: {m.eth_price:.0f} | BTC: {m.btc_price:.0f} | ETHBTC: {m.ethbtc_price:.5f}\n"
                f"RSI4H(ETH/BTC/ETHBTC): {m.eth_rsi_4h:.1f}/{m.btc_rsi_4h:.1f}/{m.ethbtc_rsi_4h:.1f}"
            )
            pushover_notify(title, msg)
            telegram_notify(f"{title}\n{msg}")

            self.recent_signals.insert(0, sig_obj)
            self.recent_signals = self.recent_signals[:50]

            db_log_signal(
                sanitize_for_json(
                    {
                        "created_at": utc_now_iso(),
                        "action": sig.action,
                        "confidence": sig.confidence,
                        "message": sig.message,
                        "meta": sig.meta,
                        "market": asdict(m),
                        "position": dict(self.position),
                    }
                )
            )

# =============================
# FastAPI + Dashboard
# =============================
bot = ETHSignalPro()
app = FastAPI(title=APP_TITLE)

db_init()

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ETHSignalPro</title>
  <style>
    body { font-family: ui-sans-serif, system-ui; margin: 18px; }
    .row { display: flex; gap: 14px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; min-width: 280px; }
    .big { font-size: 22px; font-weight: 700; }
    .muted { color: #666; }
    pre { background: #fafafa; padding: 10px; border-radius: 10px; overflow: auto; }
    button { padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; background: white; cursor: pointer; }
  </style>
</head>
<body>
  <div class="row">
    <div class="card">
      <div class="big">ETHSignalPro</div>
      <div class="muted" id="ts">loading...</div>
      <div style="margin-top:10px;">
        <button onclick="refresh()">Refresh</button>
      </div>
      <div class="muted" style="margin-top:10px;">
        Endpoints: <a href="/signals">/signals</a> | <a href="/ohlcv">/ohlcv</a> | <a href="/api/recent">/api/recent</a>
      </div>
    </div>

    <div class="card">
      <div class="big" id="action">--</div>
      <div class="muted" id="confidence">--</div>
      <div style="margin-top:10px;" id="msg">--</div>
    </div>

    <div class="card">
      <div class="big">Market</div>
      <div id="market"></div>
    </div>
  </div>

  <h3>Recent Signals</h3>
  <pre id="recent">--</pre>

  <script>
    async function refresh() {
      const r = await fetch('/signals');
      const j = await r.json();

      document.getElementById('ts').textContent = j.time || '';
      document.getElementById('action').textContent = j.signal?.action || '--';
      document.getElementById('confidence').textContent = j.signal?.confidence || '--';
      document.getElementById('msg').textContent = j.signal?.message || '--';

      const m = j.market || {};
      document.getElementById('market').innerHTML =
        `ETH: <b>${(m.eth_price||0).toFixed(0)}</b><br>` +
        `BTC: <b>${(m.btc_price||0).toFixed(0)}</b><br>` +
        `ETHBTC: <b>${(m.ethbtc_price||0).toFixed(5)}</b><br>` +
        `<span class="muted">RSI4H ETH: ${(m.eth_rsi_4h||0).toFixed(1)} | BTC: ${(m.btc_rsi_4h||0).toFixed(1)} | ETHBTC: ${(m.ethbtc_rsi_4h||0).toFixed(1)}</span><br>` +
        `<span class="muted">BB4H ETH: L ${(m.eth_bb_lower_4h||0).toFixed(0)} / M ${(m.eth_bb_mid_4h||0).toFixed(0)} / U ${(m.eth_bb_upper_4h||0).toFixed(0)}</span><br>` +
        `<span class="muted">MA200 1D ETH: ${(m.eth_ma200_1d||0).toFixed(0)}</span>`;

      const r2 = await fetch('/api/recent');
      const j2 = await r2.json();
      document.getElementById('recent').textContent = JSON.stringify(j2, null, 2);
    }
    refresh();
    setInterval(refresh, 30000);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD_HTML

@app.get("/ohlcv", response_class=JSONResponse)
def api_ohlcv():
    if not bot.cache_ohlcv:
        bot.refresh()
    return JSONResponse(content=sanitize_for_json({"time": utc_now_iso(), "data": bot.cache_ohlcv}))

@app.get("/signals", response_class=JSONResponse)
def api_signals():
    if not bot.cache_signal:
        bot.refresh()
    return JSONResponse(content=sanitize_for_json(bot.cache_signal))

@app.get("/api/recent", response_class=JSONResponse)
def api_recent():
    return JSONResponse(content=sanitize_for_json(bot.recent_signals))

@app.get("/api/health", response_class=JSONResponse)
def api_health():
    return JSONResponse(content={"ok": True, "time": utc_now_iso()})

# =============================
# Scheduler
# =============================
scheduler = BackgroundScheduler()

def _job():
    try:
        bot.refresh()
    except Exception as e:
        print(f"[refresh] error: {e}")

scheduler.add_job(_job, "interval", seconds=REFRESH_SECONDS, max_instances=1, coalesce=True)
scheduler.start()
