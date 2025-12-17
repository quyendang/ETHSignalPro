import os
import json
import time
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, Index
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================
# ENV / Config
# =========================

APP_TITLE = "ETH Loan-Safe Signals (Binance Direct) v1"

DATABASE_URL = os.environ.get("DATABASE_URL", "")
RUN_EVERY_SECONDS = int(os.environ.get("RUN_EVERY_SECONDS", "600"))  # 10 phút
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "20"))

# Anti-spam
STATE_NOTIFY_COOLDOWN_SECONDS = int(os.environ.get("STATE_NOTIFY_COOLDOWN_SECONDS", str(6 * 3600)))  # 6h
ACTION_MIN_INTERVAL_SECONDS = int(os.environ.get("ACTION_MIN_INTERVAL_SECONDS", str(3 * 3600)))      # 3h

# Manual loan config (con nhập tay)
MANUAL_LOAN_USDT = float(os.environ.get("MANUAL_LOAN_USDT", "0"))               # ví dụ 30000
MANUAL_COLLATERAL_USDT = float(os.environ.get("MANUAL_COLLATERAL_USDT", "1"))   # ví dụ 290000
MANUAL_SPOT_ETH = float(os.environ.get("MANUAL_SPOT_ETH", "0"))                 # ví dụ 138
MANUAL_AVG_ENTRY = float(os.environ.get("MANUAL_AVG_ENTRY", "0"))               # ví dụ 3150

# Risk thresholds
SAFE_LTV = float(os.environ.get("SAFE_LTV", "12"))        # %
RISK_LTV = float(os.environ.get("RISK_LTV", "15"))        # %
CRITICAL_LTV = float(os.environ.get("CRITICAL_LTV", "25"))# %

# Indicators thresholds
ETH_OVERSOLD_RSI = float(os.environ.get("ETH_OVERSOLD_RSI", "32"))
BTC_BREAKDOWN_RSI = float(os.environ.get("BTC_BREAKDOWN_RSI", "45"))
BTC_RECOVER_RSI = float(os.environ.get("BTC_RECOVER_RSI", "50"))

# Pushover
PUSHOVER_ENABLED = os.environ.get("PUSHOVER_ENABLED", "1") == "1"
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.environ.get("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.environ.get("PUSHOVER_DEVICE", "")
PUSHOVER_SOUND = os.environ.get("PUSHOVER_SOUND", "cash")

# Binance endpoints (Spot public, no key)
BINANCE_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"

# =========================
# DB setup
# =========================

Base = declarative_base()

def _normalize_pg_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url

engine = create_engine(_normalize_pg_url(DATABASE_URL), pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False) if engine else None

class BotKV(Base):
    __tablename__ = "bot_kv"
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

class SignalRow(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True, nullable=False)
    state = Column(String(16), index=True, nullable=False)       # SAFE/RISK/CRITICAL
    action = Column(String(32), index=True, nullable=False)      # HOLD/BUYBACK/REDUCE_RISK...
    confidence = Column(String(16), nullable=False)              # low/med/high
    message = Column(Text, nullable=False)
    meta_json = Column(Text, nullable=False)
    eth_price = Column(Float, nullable=True)
    btc_price = Column(Float, nullable=True)
    ethbtc_price = Column(Float, nullable=True)

Index("idx_signals_created_at", SignalRow.created_at.desc())

def init_db():
    if not engine:
        return
    Base.metadata.create_all(bind=engine)

def get_db():
    if not SessionLocal:
        # allow run without DB (still works, but no persistence)
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def kv_get(db: Optional[Session], key: str, default: str) -> str:
    if db is None:
        return default
    row = db.query(BotKV).filter(BotKV.key == key).one_or_none()
    return row.value if row else default

def kv_set(db: Optional[Session], key: str, value: str):
    if db is None:
        return
    row = db.query(BotKV).filter(BotKV.key == key).one_or_none()
    if row:
        row.value = value
        row.updated_at = dt.datetime.utcnow()
    else:
        db.add(BotKV(key=key, value=value, updated_at=dt.datetime.utcnow()))
    db.commit()

def db_add_signal(db: Optional[Session], state: str, action: str, confidence: str, message: str,
                  meta: Dict[str, Any], eth: float, btc: float, ethbtc: float):
    if db is None:
        return
    db.add(SignalRow(
        state=state,
        action=action,
        confidence=confidence,
        message=message,
        meta_json=json.dumps(meta, ensure_ascii=False),
        eth_price=eth,
        btc_price=btc,
        ethbtc_price=ethbtc
    ))
    db.commit()

# =========================
# Indicators
# =========================

def ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    out = []
    e = sum(values[:period]) / period
    out.extend([float("nan")] * (period - 1))
    out.append(e)
    for i in range(period, len(values)):
        e = values[i] * k + e * (1 - k)
        out.append(e)
    return out

def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return []
    gains = []
    losses = []
    for i in range(1, len(values)):
        ch = values[i] - values[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))

    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    out = [float("nan")] * period
    rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
    out.append(100 - (100 / (1 + rs)))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        out.append(100 - (100 / (1 + rs)))
    return out

def bbands(values: List[float], period: int = 20, stdev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    if len(values) < period:
        return [], [], []
    mid = []
    upper = []
    lower = []
    for i in range(len(values)):
        if i < period - 1:
            mid.append(float("nan"))
            upper.append(float("nan"))
            lower.append(float("nan"))
            continue
        window = values[i - period + 1:i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        sd = math.sqrt(var)
        mid.append(m)
        upper.append(m + stdev * sd)
        lower.append(m - stdev * sd)
    return lower, mid, upper

def last_finite(x: List[float]) -> Optional[float]:
    for v in reversed(x):
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        return float(v)
    return None

# =========================
# Binance fetch
# =========================

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> List[List[Any]]:
    url = f"{BINANCE_BASE}{KLINES_PATH}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def closes_from_klines(klines: List[List[Any]]) -> List[float]:
    # kline[4] = close
    return [float(k[4]) for k in klines]

# =========================
# Market + Position
# =========================

@dataclass
class Market:
    time: dt.datetime
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    btc_rsi_4h: float
    ethbtc_rsi_4h: float

    eth_bb_lower_4h: Optional[float]
    eth_bb_mid_4h: Optional[float]
    eth_bb_upper_4h: Optional[float]

    btc_ema200_4h: Optional[float]

@dataclass
class Position:
    time: dt.datetime
    loan_usdt: float
    collateral_usdt: float
    ltv_pct: float
    spot_eth: float
    avg_entry: float

def now_utc_naive() -> dt.datetime:
    return dt.datetime.utcnow()

def build_market() -> Market:
    # 4H candles
    eth_4h = fetch_klines("ETHUSDT", "4h", 220)
    btc_4h = fetch_klines("BTCUSDT", "4h", 220)
    ethbtc_4h = fetch_klines("ETHBTC", "4h", 220)

    eth_c = closes_from_klines(eth_4h)
    btc_c = closes_from_klines(btc_4h)
    ethbtc_c = closes_from_klines(ethbtc_4h)

    eth_price = eth_c[-1]
    btc_price = btc_c[-1]
    ethbtc_price = ethbtc_c[-1]

    eth_rsi_series = rsi(eth_c, 14)
    btc_rsi_series = rsi(btc_c, 14)
    ethbtc_rsi_series = rsi(ethbtc_c, 14)

    eth_rsi_4h = last_finite(eth_rsi_series) or 50.0
    btc_rsi_4h = last_finite(btc_rsi_series) or 50.0
    ethbtc_rsi_4h = last_finite(ethbtc_rsi_series) or 50.0

    bb_l, bb_m, bb_u = bbands(eth_c, 20, 2.0)
    eth_bb_lower_4h = last_finite(bb_l)
    eth_bb_mid_4h = last_finite(bb_m)
    eth_bb_upper_4h = last_finite(bb_u)

    btc_ema200 = ema(btc_c, 200)
    btc_ema200_4h = last_finite(btc_ema200)

    return Market(
        time=now_utc_naive(),
        eth_price=eth_price,
        btc_price=btc_price,
        ethbtc_price=ethbtc_price,
        eth_rsi_4h=float(eth_rsi_4h),
        btc_rsi_4h=float(btc_rsi_4h),
        ethbtc_rsi_4h=float(ethbtc_rsi_4h),
        eth_bb_lower_4h=eth_bb_lower_4h,
        eth_bb_mid_4h=eth_bb_mid_4h,
        eth_bb_upper_4h=eth_bb_upper_4h,
        btc_ema200_4h=btc_ema200_4h
    )

def build_position() -> Position:
    loan = max(MANUAL_LOAN_USDT, 0.0)
    collateral = max(MANUAL_COLLATERAL_USDT, 1.0)
    ltv = (loan / collateral) * 100.0 if collateral > 0 else 0.0
    return Position(
        time=now_utc_naive(),
        loan_usdt=loan,
        collateral_usdt=collateral,
        ltv_pct=ltv,
        spot_eth=MANUAL_SPOT_ETH,
        avg_entry=MANUAL_AVG_ENTRY
    )

# =========================
# State machine logic
# =========================

@dataclass
class Decision:
    state: str
    action: str
    confidence: str
    message: str
    meta: Dict[str, Any]

def btc_breakdown(m: Market) -> Tuple[bool, Dict[str, Any]]:
    # No hard BTC price. Use EMA200 4H + RSI.
    if m.btc_ema200_4h is None:
        return (m.btc_rsi_4h < (BTC_BREAKDOWN_RSI - 5)), {"method": "rsi_only"}
    return ((m.btc_price < m.btc_ema200_4h) and (m.btc_rsi_4h < BTC_BREAKDOWN_RSI)), {
        "method": "ema200+rsi", "btc_ema200_4h": m.btc_ema200_4h
    }

def btc_recover(m: Market) -> Tuple[bool, Dict[str, Any]]:
    if m.btc_ema200_4h is None:
        return (m.btc_rsi_4h >= BTC_RECOVER_RSI), {"method": "rsi_only"}
    return ((m.btc_price >= m.btc_ema200_4h) and (m.btc_rsi_4h >= BTC_RECOVER_RSI)), {
        "method": "ema200+rsi", "btc_ema200_4h": m.btc_ema200_4h
    }

def decide(m: Market, p: Position) -> Decision:
    loan_active = p.loan_usdt > 0
    breakdown, bmeta = btc_breakdown(m)
    recover, rmeta = btc_recover(m)

    # Base state by LTV
    if p.ltv_pct >= CRITICAL_LTV:
        state = "CRITICAL"
    elif p.ltv_pct >= RISK_LTV:
        state = "RISK"
    else:
        state = "SAFE" if p.ltv_pct <= SAFE_LTV else "RISK"

    # If loan active and BTC breakdown -> at least RISK
    if loan_active and breakdown:
        state = "CRITICAL" if p.ltv_pct >= RISK_LTV else "RISK"
    # If loan active and BTC recover and low LTV -> SAFE
    if loan_active and recover and p.ltv_pct <= SAFE_LTV:
        state = "SAFE"

    meta = {
        "ltv_pct": round(p.ltv_pct, 2),
        "loan_usdt": round(p.loan_usdt, 2),
        "collateral_usdt": round(p.collateral_usdt, 2),
        "loan_active": loan_active,
        "btc_breakdown": bool(breakdown),
        "btc_recover": bool(recover),
        "btc_rsi_4h": round(m.btc_rsi_4h, 2),
        "eth_rsi_4h": round(m.eth_rsi_4h, 2),
        "ethbtc_rsi_4h": round(m.ethbtc_rsi_4h, 2),
        **bmeta,
        **rmeta,
        "eth_bb_lower_4h": m.eth_bb_lower_4h,
        "eth_bb_mid_4h": m.eth_bb_mid_4h,
        "eth_bb_upper_4h": m.eth_bb_upper_4h,
    }

    if state == "CRITICAL":
        return Decision(
            state=state,
            action="REDUCE_RISK",
            confidence="high",
            message="CRITICAL: Đang vay + BTC breakdown hoặc LTV cao → ưu tiên GIẢM NỢ/GIẢM RỦI RO, không tăng vị thế.",
            meta=meta
        )

    if state == "RISK":
        if loan_active and breakdown:
            return Decision(
                state=state,
                action="REDUCE_RISK",
                confidence="high",
                message="RISK: BTC breakdown trong khi đang vay → ưu tiên giảm rủi ro/giảm nợ, KHÔNG tăng vị thế.",
                meta=meta
            )

        # BUYBACK chỉ là “mua lại/nhặt” mức nhỏ, ưu tiên vốn có sẵn (không tăng vay)
        # Condition: ETH oversold + chạm/ dưới BB lower
        bb_ok = (m.eth_bb_lower_4h is not None) and (m.eth_price <= m.eth_bb_lower_4h * 1.01)
        if (not breakdown) and (m.eth_rsi_4h <= ETH_OVERSOLD_RSI) and bb_ok:
            return Decision(
                state=state,
                action="BUYBACK",
                confidence="med",
                message="RISK: ETH oversold + gần BB lower và BTC chưa breakdown → có thể BUYBACK nhẹ (không tăng vay).",
                meta=meta
            )

        return Decision(
            state=state,
            action="HOLD",
            confidence="med",
            message="RISK: Giữ nguyên. Không tăng vay. Chờ BTC hồi cấu trúc (reclaim EMA200 4H + RSI) rồi mới tính.",
            meta=meta
        )

    # SAFE
    return Decision(
        state=state,
        action="SAFE_HOLD",
        confidence="med",
        message="SAFE: LTV thấp và BTC không xấu → giữ/quan sát. Nếu muốn vay thêm thì tăng rất nhỏ, đặt giới hạn LTV.",
        meta=meta
    )

# =========================
# Notify + Anti-spam
# =========================

def pushover_notify(title: str, message: str):
    if not PUSHOVER_ENABLED:
        return
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return
    data = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": 0,
        "sound": PUSHOVER_SOUND or "cash",
    }
    if PUSHOVER_DEVICE:
        data["device"] = PUSHOVER_DEVICE
    try:
        requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=15)
    except Exception:
        pass

def should_notify(db: Optional[Session], d: Decision) -> Tuple[bool, str]:
    last_state = kv_get(db, "last_state", "UNKNOWN")
    last_action = kv_get(db, "last_action", "NONE")

    last_notify_ts = float(kv_get(db, "last_notify_ts", "0"))
    last_action_ts = float(kv_get(db, "last_action_ts", "0"))
    now_ts = time.time()

    state_changed = d.state != last_state
    action_changed = d.action != last_action

    if state_changed:
        return True, "state_changed"
    if action_changed and (now_ts - last_action_ts >= 60):
        return True, "action_changed"
    if now_ts - last_notify_ts >= STATE_NOTIFY_COOLDOWN_SECONDS:
        return True, "state_cooldown"
    if now_ts - last_action_ts >= ACTION_MIN_INTERVAL_SECONDS:
        return True, "action_min_interval"
    return False, "no_change"

def persist_and_notify(db: Optional[Session], m: Market, p: Position, d: Decision, reason: str):
    # Persist
    db_add_signal(db, d.state, d.action, d.confidence, d.message, d.meta, m.eth_price, m.btc_price, m.ethbtc_price)
    kv_set(db, "last_state", d.state)
    kv_set(db, "last_action", d.action)
    kv_set(db, "last_action_ts", str(time.time()))
    kv_set(db, "last_notify_ts", str(time.time()))

    # Notify
    title = f"{d.action} | {d.state}"
    msg = (
        f"{d.message}\n\n"
        f"ETH: {m.eth_price:.2f}\n"
        f"BTC: {m.btc_price:.2f}\n"
        f"ETHBTC: {m.ethbtc_price:.5f}\n"
        f"LTV: {p.ltv_pct:.2f}% (loan {p.loan_usdt:.0f} / collateral {p.collateral_usdt:.0f})\n"
        f"Reason: {reason}\n"
        f"Time: {m.time.isoformat()}Z"
    )
    pushover_notify(title, msg)

# =========================
# FastAPI app + background loop
# =========================

app = FastAPI(title=APP_TITLE)

@app.on_event("startup")
def on_startup():
    init_db()
    # start loop in background thread style (simple)
    import threading
    t = threading.Thread(target=run_loop_forever, daemon=True)
    t.start()

def run_loop_forever():
    while True:
        db = None
        try:
            db = SessionLocal() if SessionLocal else None
            m = build_market()
            p = build_position()
            d = decide(m, p)
            notify, reason = should_notify(db, d)

            if notify:
                persist_and_notify(db, m, p, d, reason)

            # Always update last_market cache (kv) for /api/state to display even without notify
            cache = {
                "time": m.time.isoformat() + "Z",
                "eth_price": m.eth_price,
                "btc_price": m.btc_price,
                "ethbtc_price": m.ethbtc_price,
                "eth_rsi_4h": m.eth_rsi_4h,
                "btc_rsi_4h": m.btc_rsi_4h,
                "ethbtc_rsi_4h": m.ethbtc_rsi_4h,
                "eth_bb_lower_4h": m.eth_bb_lower_4h,
                "eth_bb_mid_4h": m.eth_bb_mid_4h,
                "eth_bb_upper_4h": m.eth_bb_upper_4h,
                "btc_ema200_4h": m.btc_ema200_4h,
            }
            if db is not None:
                kv_set(db, "last_market_json", json.dumps(cache))

        except Exception as e:
            # Error notify max 1h
            try:
                if db is None and SessionLocal:
                    db = SessionLocal()
                last_err_ts = float(kv_get(db, "last_err_ts", "0"))
                if time.time() - last_err_ts > 3600:
                    kv_set(db, "last_err_ts", str(time.time()))
                    pushover_notify("BOT ERROR", f"{type(e).__name__}: {str(e)[:800]}")
            except Exception:
                pass
        finally:
            try:
                if db is not None:
                    db.close()
            except Exception:
                pass

        time.sleep(RUN_EVERY_SECONDS)

# =========================
# API
# =========================

@app.get("/api/state")
def api_state(db: Optional[Session] = Depends(get_db)):
    last_market = kv_get(db, "last_market_json", "{}")
    try:
        last_market_obj = json.loads(last_market)
    except Exception:
        last_market_obj = {}

    return {
        "ok": True,
        "config": {
            "RUN_EVERY_SECONDS": RUN_EVERY_SECONDS,
            "STATE_NOTIFY_COOLDOWN_SECONDS": STATE_NOTIFY_COOLDOWN_SECONDS,
            "ACTION_MIN_INTERVAL_SECONDS": ACTION_MIN_INTERVAL_SECONDS,
            "SAFE_LTV": SAFE_LTV,
            "RISK_LTV": RISK_LTV,
            "CRITICAL_LTV": CRITICAL_LTV,
            "ETH_OVERSOLD_RSI": ETH_OVERSOLD_RSI,
            "BTC_BREAKDOWN_RSI": BTC_BREAKDOWN_RSI,
            "BTC_RECOVER_RSI": BTC_RECOVER_RSI,
            "MANUAL_LOAN_USDT": MANUAL_LOAN_USDT,
            "MANUAL_COLLATERAL_USDT": MANUAL_COLLATERAL_USDT,
            "MANUAL_SPOT_ETH": MANUAL_SPOT_ETH,
            "MANUAL_AVG_ENTRY": MANUAL_AVG_ENTRY,
        },
        "runtime": {
            "last_state": kv_get(db, "last_state", "UNKNOWN"),
            "last_action": kv_get(db, "last_action", "NONE"),
            "last_notify_ts": kv_get(db, "last_notify_ts", "0"),
            "last_action_ts": kv_get(db, "last_action_ts", "0"),
        },
        "market": last_market_obj,
    }

@app.get("/api/signals")
def api_signals(limit: int = 100, db: Optional[Session] = Depends(get_db)):
    if db is None:
        return {"ok": True, "signals": [], "note": "DB disabled (DATABASE_URL not set)"}
    limit = max(1, min(limit, 500))
    rows = db.query(SignalRow).order_by(SignalRow.created_at.desc()).limit(limit).all()
    out = []
    for r in rows:
        out.append({
            "time": r.created_at.isoformat() + "Z",
            "state": r.state,
            "action": r.action,
            "confidence": r.confidence,
            "message": r.message,
            "meta": json.loads(r.meta_json),
            "eth_price": r.eth_price,
            "btc_price": r.btc_price,
            "ethbtc_price": r.ethbtc_price,
        })
    return {"ok": True, "signals": out}

@app.get("/", response_class=HTMLResponse)
def dashboard(db: Optional[Session] = Depends(get_db)):
    latest = None
    signals = []
    if db is not None:
        latest = db.query(SignalRow).order_by(SignalRow.created_at.desc()).first()
        signals = db.query(SignalRow).order_by(SignalRow.created_at.desc()).limit(50).all()

    last_market = {}
    try:
        last_market = json.loads(kv_get(db, "last_market_json", "{}")) if db is not None else {}
    except Exception:
        last_market = {}

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    latest_html = "<div class='muted'>No signals yet.</div>"
    if latest:
        latest_html = f"""
        <div class="card">
          <div class="h">Latest</div>
          <div><b>{esc(latest.action)}</b> | <b>{esc(latest.state)}</b> | {latest.created_at} UTC</div>
          <div class="msg">{esc(latest.message)}</div>
          <pre class="pre">{esc(latest.meta_json)}</pre>
        </div>
        """

    rows = ""
    for s in signals:
        rows += f"""
        <tr>
          <td>{s.created_at}</td>
          <td><b>{esc(s.action)}</b></td>
          <td>{esc(s.state)}</td>
          <td>{esc(s.confidence)}</td>
          <td>{esc(s.message)[:140]}</td>
          <td>{s.eth_price or ""}</td>
          <td>{s.btc_price or ""}</td>
          <td>{s.ethbtc_price or ""}</td>
        </tr>
        """

    cfg = {
        "RUN_EVERY_SECONDS": RUN_EVERY_SECONDS,
        "SAFE_LTV": SAFE_LTV, "RISK_LTV": RISK_LTV, "CRITICAL_LTV": CRITICAL_LTV,
        "ETH_OVERSOLD_RSI": ETH_OVERSOLD_RSI,
        "BTC_BREAKDOWN_RSI": BTC_BREAKDOWN_RSI, "BTC_RECOVER_RSI": BTC_RECOVER_RSI,
        "MANUAL_LOAN_USDT": MANUAL_LOAN_USDT,
        "MANUAL_COLLATERAL_USDT": MANUAL_COLLATERAL_USDT,
        "MANUAL_SPOT_ETH": MANUAL_SPOT_ETH,
        "MANUAL_AVG_ENTRY": MANUAL_AVG_ENTRY,
        "PUSHOVER_ENABLED": PUSHOVER_ENABLED,
    }

    return f"""
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{APP_TITLE}</title>

    <style>
      :root{
        --bg:#f6f8fc;
        --card:#ffffff;
        --text:#0f172a;
        --muted:#64748b;
        --border:#e5eaf3;
        --soft:#f1f5ff;
        --codebg:#f8fafc;
        --shadow: 0 10px 30px rgba(15,23,42,.08);
        --shadow2: 0 6px 18px rgba(15,23,42,.06);
        --radius: 18px;

        --blue:#2563eb;
        --blueSoft:#e8f0ff;

        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      }

      *{ box-sizing:border-box; }
      body{
        margin:0;
        padding:24px;
        font-family: var(--sans);
        background: radial-gradient(1200px 600px at 20% 0%, #eaf2ff 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 20%, #eef2ff 0%, transparent 55%),
                    var(--bg);
        color:var(--text);
      }

      /* container */
      .wrap{ max-width: 1200px; margin: 0 auto; }

      /* top header */
      .topbar{
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:16px;
        margin-bottom:16px;
      }
      .title{
        font-size:20px;
        font-weight:800;
        letter-spacing:.2px;
      }
      .subtitle{
        margin-top:4px;
        color:var(--muted);
        font-size:13px;
      }
      .chips{ display:flex; gap:10px; flex-wrap:wrap; }
      .chip{
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:8px 12px;
        background: var(--card);
        border:1px solid var(--border);
        border-radius:999px;
        box-shadow: var(--shadow2);
        font-size:13px;
        color:var(--muted);
        text-decoration:none;
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
      }
      .chip:hover{
        transform: translateY(-1px);
        box-shadow: var(--shadow);
        border-color:#d7deea;
      }
      .dot{
        width:8px; height:8px; border-radius:999px;
        background: var(--blue);
        box-shadow: 0 0 0 4px var(--blueSoft);
      }

      .row{
        display:flex;
        gap:16px;
        flex-wrap:wrap;
      }

      .card{
        background:var(--card);
        border:1px solid var(--border);
        border-radius: var(--radius);
        padding:16px;
        min-width:320px;
        flex:1;
        box-shadow: var(--shadow2);
      }

      .h{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        font-size:14px;
        font-weight:800;
        margin-bottom:10px;
        color:var(--text);
      }

      .muted{ color:var(--muted); font-size:13px; line-height:1.4; }

      .pre{
        background: var(--codebg);
        border:1px solid var(--border);
        border-radius: 14px;
        padding:12px 12px;
        overflow:auto;
        font-family: var(--mono);
        font-size:12px;
        line-height:1.55;
        color:#0b1220;
      }

      code{
        font-family: var(--mono);
        background: #eef2ff;
        border:1px solid #e5e7ff;
        padding:2px 6px;
        border-radius:999px;
        color:#1e40af;
        font-size:12px;
      }

      a{
        color: var(--blue);
        text-decoration: none;
        font-weight:600;
      }
      a:hover{ text-decoration: underline; }

      /* table */
      table{
        width:100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top:12px;
        overflow:hidden;
        border:1px solid var(--border);
        border-radius: 16px;
        background: var(--card);
      }
      thead th{
        text-align:left;
        color: var(--muted);
        font-weight:800;
        font-size:12px;
        padding:12px 12px;
        background: linear-gradient(180deg, #fbfcff 0%, #f7f9ff 100%);
        border-bottom:1px solid var(--border);
        position: sticky;
        top: 0;
        z-index: 1;
      }
      tbody td{
        padding:12px 12px;
        font-size:13px;
        vertical-align:top;
        border-bottom:1px solid var(--border);
      }
      tbody tr:hover td{
        background: #f8fbff;
      }
      tbody tr:last-child td{ border-bottom:none; }

      /* small helpers */
      .spacer{ height:16px; }
      .card.full{ margin-top:16px; }
      .right{ text-align:right; }

      /* mobile */
      @media (max-width: 720px){
        body{ padding:16px; }
        .card{ min-width: 100%; }
        .topbar{ align-items:flex-start; flex-direction:column; }
      }
    </style>
  </head>

  <body>
    <div class="wrap">
      <div class="topbar">
        <div>
          <div class="title">{APP_TITLE}</div>
          <div class="subtitle">Dashboard • FastAPI render • Flat & bright UI</div>
        </div>
        <div class="chips">
          <a class="chip" href="/api/state"><span class="dot"></span> /api/state</a>
          <a class="chip" href="/api/signals?limit=100"><span class="dot"></span> /api/signals</a>
        </div>
      </div>

      <div class="row">
        <div class="card">
          <div class="h">
            <span>Market (last)</span>
            <span class="muted">Live snapshot</span>
          </div>
          <pre class="pre">{esc(json.dumps(last_market, indent=2, ensure_ascii=False))}</pre>
          <div class="muted" style="margin-top:10px">
            Endpoints: <a href="/api/state">/api/state</a> · <a href="/api/signals?limit=100">/api/signals</a>
          </div>
        </div>

        <div class="card">
          <div class="h">
            <span>Config</span>
            <span class="muted">Runtime</span>
          </div>
          <pre class="pre">{esc(json.dumps(cfg, indent=2, ensure_ascii=False))}</pre>
          <div class="muted" style="margin-top:10px">
            Binance: Spot public klines (ETHUSDT/BTCUSDT/ETHBTC 4h)
          </div>
        </div>

        {latest_html}
      </div>

      <div class="card full">
        <div class="h">
          <span>Recent signals</span>
          <span class="muted">Latest events</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Action</th>
              <th>State</th>
              <th>Conf</th>
              <th>Message</th>
              <th>ETH</th>
              <th>BTC</th>
              <th>ETHBTC</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
  </body>
</html>
    """

