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

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "").strip()
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "").strip()
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "signals").strip()

SPOT_ETH = float(os.getenv("SPOT_ETH", "0") or 0)
AVG_ENTRY = float(os.getenv("AVG_ENTRY", "0") or 0)

MAX_LOAN_USDT = float(os.getenv("MAX_LOAN_USDT", "0") or 0)
MAX_LOAN_PCT_NAV = float(os.getenv("MAX_LOAN_PCT_NAV", "0.25") or 0.25)

SELL_COOLDOWN_SECONDS = int(os.getenv("SELL_COOLDOWN_SECONDS", "14400"))

# =============================
# Utilities
# =============================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sanitize_for_json(x: Any) -> Any:
    # convert NaN/Inf to None recursively
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
    """
    Uses Bybit v5 market kline & tickers (public).
    """
    BASE = "https://api.bybit.com"

    # Map our interval -> Bybit interval
    # Bybit kline interval uses minutes: 1,3,5,15,30,60,120,240,360,720,"D","W","M"
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
        """
        Returns list of candles sorted ascending time:
        {timestamp, open, high, low, close, volume}
        """
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

        # Bybit: result.list is array of arrays: [startTime, open, high, low, close, volume, turnover]
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

        # Bybit returns newest->oldest; we reverse to ascending
        candles.sort(key=lambda x: x["timestamp"])
        return candles

    def get_last_price(self, symbol: str) -> float:
        data = self._get(
            "/v5/market/tickers",
            {"category": "spot", "symbol": symbol},
        )
        items = data.get("result", {}).get("list", [])
        if not items:
            return float("nan")
        return float(items[0].get("lastPrice", "nan"))

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
    action: str  # HOLD / SELL_ROTATE / BUYBACK / LOAN_VALUE / LOAN_MOMENTUM / REPAY
    confidence: str  # low/med/high
    message: str
    meta: Dict[str, Any]

class StrategyEngine:
    """
    “Thầy” version: rules are explicit and consistent:
    - Rotation: sell some ETH at resistance, buy back at support
    - Loan: split VALUE vs MOMENTUM (đã sửa lỗi logic)
    - Cooldown sell 4h
    """

    def __init__(self):
        self.last_sell_ts: Optional[int] = None

    def _in_cooldown(self, now_ts: int) -> bool:
        if self.last_sell_ts is None:
            return False
        return (now_ts - self.last_sell_ts) < SELL_COOLDOWN_SECONDS

    def decide(self, m: MarketSnapshot, position: Dict[str, Any]) -> Signal:
        # Position
        spot_eth = float(position.get("spot_eth", SPOT_ETH))
        loan_usdt = float(position.get("loan_usdt", 0.0))
        avg_entry = float(position.get("avg_entry", AVG_ENTRY))

        now_ts = int(time.time())

        # Zones (dynamic-ish)
        # Use BB 4H + MA200 1D as anchors
        buy_zone1 = (m.eth_bb_lower_4h, m.eth_bb_mid_4h)  # mean reversion area
        # Safety floor: BB lower as "value-ish" reference
        # Rotation sell zone: near BB upper or key resistance levels
        sell_zone = (m.eth_bb_upper_4h * 0.995, m.eth_bb_upper_4h * 1.02)

        # Additional hard bands (you can tune later)
        # These keep it stable across months
        hard_sell1 = (3350, 3450)
        hard_sell2 = (3550, 3700)
        hard_buy1 = (3000, 3050)
        hard_buy2 = (2850, 2920)

        # ==========
        # 1) Repay logic if currently in loan & market gives rebound
        # ==========
        if loan_usdt > 0:
            # simple repay trigger: ETH back above BB mid and RSI>55
            if (m.eth_price > m.eth_bb_mid_4h) and (m.eth_rsi_4h >= 55):
                return Signal(
                    action="REPAY",
                    confidence="med",
                    message=f"ETH hồi về vùng trả nợ: giá {m.eth_price:.0f} > BB_mid {m.eth_bb_mid_4h:.0f}, RSI4H {m.eth_rsi_4h:.1f}. Ưu tiên chốt 1 phần để giảm áp lực vay.",
                    meta={
                        "suggested_sell_eth": round(min(spot_eth * 0.08, 10), 3),
                        "loan_usdt": loan_usdt,
                        "eth_price": m.eth_price,
                    },
                )

        # ==========
        # 2) Rotation SELL (sell some ETH to buy back lower)
        # Conditions: ETHBTC weak/neutral + ETH at resistance
        # ==========
        if not self._in_cooldown(now_ts):
            eth_at_resistance = (hard_sell1[0] <= m.eth_price <= hard_sell1[1]) or (hard_sell2[0] <= m.eth_price <= hard_sell2[1]) or (sell_zone[0] <= m.eth_price <= sell_zone[1])
            ethbtc_not_strong = (m.ethbtc_rsi_4h <= 55) or (m.ethbtc_macd_hist_4h < 0)

            if eth_at_resistance and ethbtc_not_strong and spot_eth > 0:
                # Sell size: 20–30 ETH if you have big stack; else 20% position
                suggested = 0.0
                if spot_eth >= 120:
                    suggested = 30
                elif spot_eth >= 80:
                    suggested = 20
                else:
                    suggested = spot_eth * 0.2

                self.last_sell_ts = now_ts
                return Signal(
                    action="SELL_ROTATE",
                    confidence="med",
                    message=f"Vùng bán xoay vòng: ETH {m.eth_price:.0f} chạm kháng cự; ETHBTC chưa mạnh. Bán một phần để chờ mua lại thấp.",
                    meta={
                        "sell_eth": round(suggested, 3),
                        "sell_zone": {"hard1": hard_sell1, "hard2": hard_sell2, "bb_zone": (round(sell_zone[0]), round(sell_zone[1]))},
                        "ethbtc": m.ethbtc_price,
                        "ethbtc_rsi_4h": m.ethbtc_rsi_4h,
                    },
                )

        # ==========
        # 3) BUYBACK (re-buy after pullback)
        # ==========
        buyback_ok = (hard_buy1[0] <= m.eth_price <= hard_buy1[1]) or (hard_buy2[0] <= m.eth_price <= hard_buy2[1]) or (m.eth_price <= m.eth_bb_lower_4h * 1.01)
        if buyback_ok:
            return Signal(
                action="BUYBACK",
                confidence="med",
                message=f"Vùng mua lại: ETH {m.eth_price:.0f} vào hỗ trợ/BB lower. Ưu tiên mua lại phần đã bán trước đó.",
                meta={
                    "buy_zones": {"hard1": hard_buy1, "hard2": hard_buy2, "bb_lower": round(m.eth_bb_lower_4h)},
                    "eth_rsi_4h": m.eth_rsi_4h,
                    "btc_price": m.btc_price,
                },
            )

        # ==========
        # 4) LOAN VALUE (đã sửa logic): vay khi ETH rẻ + BTC chưa breakdown + ETHBTC không gãy đáy
        # ==========
        # Note: BTC safety threshold dynamic-ish: if RSI1D too weak, be strict
        btc_floor = 88000 if m.btc_rsi_1d > 35 else 90000
        eth_value_zone = (m.eth_price <= 3000) or (m.eth_price <= m.eth_bb_lower_4h * 1.01)

        ethbtc_ok = (m.ethbtc_price >= 0.0325)  # don't require >0.0345 here!
        btc_ok = (m.btc_price >= btc_floor)

        if loan_usdt <= 0 and MAX_LOAN_USDT > 0 and eth_value_zone and btc_ok and ethbtc_ok:
            # loan size: capped by NAV and config
            nav_usdt = spot_eth * m.eth_price
            cap_by_nav = nav_usdt * MAX_LOAN_PCT_NAV
            loan_amt = max(0, min(MAX_LOAN_USDT, cap_by_nav))
            # Keep it conservative in VALUE mode
            loan_amt *= 0.6

            return Signal(
                action="LOAN_VALUE",
                confidence="med",
                message=f"VAY VALUE: ETH đang rẻ ({m.eth_price:.0f}), BTC không breakdown (>= {btc_floor:.0f}), ETHBTC không gãy đáy. Vay nhỏ để mua thêm ETH.",
                meta={
                    "loan_usdt_suggested": round(loan_amt, 2),
                    "btc_floor": btc_floor,
                    "ethbtc": m.ethbtc_price,
                    "nav_usdt": round(nav_usdt, 2),
                    "max_pct_nav": MAX_LOAN_PCT_NAV,
                },
            )

        # ==========
        # 5) LOAN MOMENTUM: vay khi ETHBTC breakout + BTC ổn + ETH đang reclaim
        # ==========
        momentum_ok = (m.ethbtc_price > 0.0345) and (m.ethbtc_rsi_4h >= 55) and (m.btc_price >= 90000) and (m.eth_price >= m.eth_bb_mid_4h)
        if loan_usdt <= 0 and MAX_LOAN_USDT > 0 and momentum_ok:
            nav_usdt = spot_eth * m.eth_price
            cap_by_nav = nav_usdt * MAX_LOAN_PCT_NAV
            loan_amt = max(0, min(MAX_LOAN_USDT, cap_by_nav))
            # smaller than value sometimes; momentum is “follow-through”
            loan_amt *= 0.5

            return Signal(
                action="LOAN_MOMENTUM",
                confidence="high",
                message="VAY MOMENTUM: ETHBTC breakout > 0.0345 và ETH reclaim BB mid, BTC ổn. Có thể vay nhỏ để theo sóng ETH.",
                meta={
                    "loan_usdt_suggested": round(loan_amt, 2),
                    "ethbtc": m.ethbtc_price,
                    "eth_price": m.eth_price,
                    "btc_price": m.btc_price,
                },
            )

        # Default HOLD
        return Signal(
            action="HOLD",
            confidence="low",
            message="Chưa có tín hiệu rõ ràng. Ưu tiên giữ và chờ điểm đẹp (support/resistance hoặc ETHBTC breakout).",
            meta={
                "eth_price": m.eth_price,
                "btc_price": m.btc_price,
                "ethbtc": m.ethbtc_price,
            },
        )

# =============================
# Notifications + Supabase
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

def supabase_log_signal(payload: Dict[str, Any]):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    # Supabase REST: POST /rest/v1/{table}
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        httpx.post(url, headers=headers, json=payload, timeout=20.0)
    except Exception:
        pass

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

        # indicators
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
        # Fetch all pairs & tfs
        pairs = {
            "ethusdt": "ETHUSDT",
            "btcusdt": "BTCUSDT",
            "ethbtc": "ETHBTC",
        }
        tfs = ["4h", "1d"]

        o: Dict[str, Any] = {}
        for k, sym in pairs.items():
            o[k] = {}
            for tf in tfs:
                o[k][tf] = self.build_pair_tf(sym, tf, limit=300)

        # Build market snapshot for strategy
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

        # Save caches
        self.cache_ohlcv = o
        self.cache_market = m
        self.cache_signal = sig_obj

        # Notify only when action != HOLD
        if sig.action != "HOLD":
            title = f"[ETHSignalPro] {sig.action} ({sig.confidence})"
            msg = sig.message + "\n" + f"ETH: {m.eth_price:.0f} | BTC: {m.btc_price:.0f} | ETHBTC: {m.ethbtc_price:.5f}"
            pushover_notify(title, msg)
            telegram_notify(f"{title}\n{msg}")

            # Keep recent signals
            self.recent_signals.insert(0, sig_obj)
            self.recent_signals = self.recent_signals[:50]

            # Supabase log
            supabase_log_signal(
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
        `<span class="muted">RSI4H ETH: ${(m.eth_rsi_4h||0).toFixed(1)} | BTC: ${(m.btc_rsi_4h||0).toFixed(1)} | ETHBTC: ${(m.ethbtc_rsi_4h||0).toFixed(1)}</span>`;

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
    # Serve cached data; if empty -> refresh now
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
        # Don't crash scheduler
        print(f"[refresh] error: {e}")

scheduler.add_job(_job, "interval", seconds=REFRESH_SECONDS, max_instances=1, coalesce=True)
scheduler.start()

# NOTE: run with uvicorn:
# uvicorn app:app --host 0.0.0.0 --port 8000
