import os
import json
import time
import math
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, Index
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session


# =========================
# Config
# =========================

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgres://koyeb-adm:*******@ep-aged-bush-a1op42oy.ap-southeast-1.pg.koyeb.app/koyebdb"
)

# Bot cadence (anti spam primary): 10 minutes
RUN_EVERY_SECONDS = int(os.environ.get("RUN_EVERY_SECONDS", str(10 * 60)))

# State notification cooldown: default 6 hours
STATE_NOTIFY_COOLDOWN_SECONDS = int(os.environ.get("STATE_NOTIFY_COOLDOWN_SECONDS", str(6 * 3600)))

# If the exact same ACTION repeats, do not notify more often than this (extra guard)
ACTION_MIN_INTERVAL_SECONDS = int(os.environ.get("ACTION_MIN_INTERVAL_SECONDS", str(3 * 3600)))

# Pushover
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.environ.get("PUSHOVER_USER", "")
PUSHOVER_ENABLED = os.environ.get("PUSHOVER_ENABLED", "1") == "1"

# Optional: market data fetch URL (if you have your own service)
# Your service should return the same shape as MarketSnapshot below.
MARKET_ENDPOINT = os.environ.get("MARKET_ENDPOINT", "")  # e.g. https://your-api/market

# Optional: position endpoint (loan, collateral, etc.)
POSITION_ENDPOINT = os.environ.get("POSITION_ENDPOINT", "")  # e.g. https://your-api/position

# Safety thresholds (NOT hard BTC price; avoid fixed like 88k)
# We'll use structure checks instead: ema200_4h reclaim/breakdown etc.
CRITICAL_LTV = float(os.environ.get("CRITICAL_LTV", "25"))  # %
RISK_LTV = float(os.environ.get("RISK_LTV", "15"))          # %
SAFE_LTV = float(os.environ.get("SAFE_LTV", "12"))          # %

# BTC breakdown heuristic: breakdown if btc_price < btc_ema200_4h AND btc_rsi_4h < 45
BTC_BREAKDOWN_RSI = float(os.environ.get("BTC_BREAKDOWN_RSI", "45"))
BTC_RECOVER_RSI = float(os.environ.get("BTC_RECOVER_RSI", "50"))

# ETH add/avoid heuristics (no hard BTC price)
ETH_OVERSOLD_RSI = float(os.environ.get("ETH_OVERSOLD_RSI", "32"))

APP_TITLE = "Loan-Safe Bot v3 (State Machine)"


# =========================
# DB setup
# =========================

Base = declarative_base()

def _normalize_pg_url(url: str) -> str:
    # SQLAlchemy wants postgresql:// not postgres:// in some setups
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url

engine = create_engine(_normalize_pg_url(DATABASE_URL), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class BotState(Base):
    __tablename__ = "bot_state"
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class SignalRow(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True, nullable=False)
    state = Column(String(16), index=True, nullable=False)       # SAFE/RISK/CRITICAL
    action = Column(String(32), index=True, nullable=False)      # REDUCE_RISK / HOLD / ...
    confidence = Column(String(16), nullable=False)              # low/med/high
    message = Column(Text, nullable=False)
    meta_json = Column(Text, nullable=False)                     # JSON dump
    eth_price = Column(Float, nullable=True)
    btc_price = Column(Float, nullable=True)
    ethbtc_price = Column(Float, nullable=True)

Index("idx_signals_created_at", SignalRow.created_at.desc())


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# Models
# =========================

@dataclass
class MarketSnapshot:
    time: dt.datetime
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    btc_rsi_4h: float
    ethbtc_rsi_4h: float

    eth_bb_lower_4h: Optional[float] = None
    eth_bb_mid_4h: Optional[float] = None
    eth_bb_upper_4h: Optional[float] = None

    btc_ema200_4h: Optional[float] = None
    btc_ema89_4h: Optional[float] = None
    btc_ema34_4h: Optional[float] = None


@dataclass
class PositionSnapshot:
    time: dt.datetime
    # loan / collateral tracking
    loan_usdt: float
    collateral_usdt: float
    ltv_pct: float

    spot_eth: float = 0.0
    avg_entry: float = 0.0


@dataclass
class Decision:
    state: str               # SAFE / RISK / CRITICAL
    action: str              # HOLD / REDUCE_RISK / ...
    confidence: str          # low/med/high
    message: str
    meta: Dict[str, Any]


# =========================
# Utilities
# =========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def db_get_state(db: Session, key: str, default: str) -> str:
    row = db.query(BotState).filter(BotState.key == key).one_or_none()
    return row.value if row else default


def db_set_state(db: Session, key: str, value: str):
    row = db.query(BotState).filter(BotState.key == key).one_or_none()
    if row:
        row.value = value
        row.updated_at = dt.datetime.utcnow()
    else:
        row = BotState(key=key, value=value, updated_at=dt.datetime.utcnow())
        db.add(row)
    db.commit()


def pushover_send(title: str, message: str, priority: int = 0):
    if not PUSHOVER_ENABLED:
        return
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            timeout=10,
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "title": title[:100],
                "message": message[:1024],
                "priority": priority,
            },
        )
    except Exception:
        # never crash bot because of push
        pass


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Fetch market & position
# =========================

def fetch_market() -> MarketSnapshot:
    """
    You can:
    - Plug MARKET_ENDPOINT to your own API, OR
    - Replace this function to pull from Binance, etc.
    """
    if not MARKET_ENDPOINT:
        raise RuntimeError("MARKET_ENDPOINT is empty. Set it to your market snapshot API.")
    r = requests.get(MARKET_ENDPOINT, timeout=15)
    r.raise_for_status()
    d = r.json()

    # Expect keys similar to your logs:
    # d = { eth_price, btc_price, ethbtc_price, eth_rsi_4h, btc_rsi_4h, ethbtc_rsi_4h, eth_bb_lower_4h, btc_ema200_4h, ... }
    return MarketSnapshot(
        time=now_utc(),
        eth_price=safe_float(d.get("eth_price")),
        btc_price=safe_float(d.get("btc_price")),
        ethbtc_price=safe_float(d.get("ethbtc_price")),
        eth_rsi_4h=safe_float(d.get("eth_rsi_4h")),
        btc_rsi_4h=safe_float(d.get("btc_rsi_4h")),
        ethbtc_rsi_4h=safe_float(d.get("ethbtc_rsi_4h")),
        eth_bb_lower_4h=(safe_float(d.get("eth_bb_lower_4h")) if d.get("eth_bb_lower_4h") is not None else None),
        eth_bb_mid_4h=(safe_float(d.get("eth_bb_mid_4h")) if d.get("eth_bb_mid_4h") is not None else None),
        eth_bb_upper_4h=(safe_float(d.get("eth_bb_upper_4h")) if d.get("eth_bb_upper_4h") is not None else None),
        btc_ema200_4h=(safe_float(d.get("btc_ema200_4h")) if d.get("btc_ema200_4h") is not None else None),
        btc_ema89_4h=(safe_float(d.get("btc_ema89_4h")) if d.get("btc_ema89_4h") is not None else None),
        btc_ema34_4h=(safe_float(d.get("btc_ema34_4h")) if d.get("btc_ema34_4h") is not None else None),
    )


def fetch_position() -> PositionSnapshot:
    """
    If POSITION_ENDPOINT is not set, we default to a "no loan" position.
    Your endpoint should return:
      { loan_usdt, collateral_usdt, ltv_pct, spot_eth, avg_entry }
    """
    if not POSITION_ENDPOINT:
        return PositionSnapshot(
            time=now_utc(),
            loan_usdt=0.0,
            collateral_usdt=1.0,
            ltv_pct=0.0,
            spot_eth=0.0,
            avg_entry=0.0,
        )
    r = requests.get(POSITION_ENDPOINT, timeout=15)
    r.raise_for_status()
    d = r.json()
    return PositionSnapshot(
        time=now_utc(),
        loan_usdt=safe_float(d.get("loan_usdt")),
        collateral_usdt=max(safe_float(d.get("collateral_usdt")), 1.0),
        ltv_pct=safe_float(d.get("ltv_pct")),
        spot_eth=safe_float(d.get("spot_eth", 0.0)),
        avg_entry=safe_float(d.get("avg_entry", 0.0)),
    )


# =========================
# Core logic: state machine
# =========================

def is_btc_breakdown(m: MarketSnapshot) -> Tuple[bool, Dict[str, Any]]:
    """
    No hard 'BTC > 90k' rule.
    Use structure: price vs EMA200 4H + RSI filter.
    """
    meta = {}
    if m.btc_ema200_4h is None:
        # fallback: RSI-only (weaker)
        breakdown = m.btc_rsi_4h < (BTC_BREAKDOWN_RSI - 5)
        meta["method"] = "rsi_only"
        meta["btc_ema200_4h"] = None
    else:
        breakdown = (m.btc_price < m.btc_ema200_4h) and (m.btc_rsi_4h < BTC_BREAKDOWN_RSI)
        meta["method"] = "ema200+rsi"
        meta["btc_ema200_4h"] = m.btc_ema200_4h
    meta["btc_price"] = m.btc_price
    meta["btc_rsi_4h"] = m.btc_rsi_4h
    meta["btc_breakdown"] = breakdown
    return breakdown, meta


def is_btc_recover(m: MarketSnapshot) -> Tuple[bool, Dict[str, Any]]:
    meta = {}
    if m.btc_ema200_4h is None:
        recover = m.btc_rsi_4h > BTC_RECOVER_RSI
        meta["method"] = "rsi_only"
    else:
        recover = (m.btc_price >= m.btc_ema200_4h) and (m.btc_rsi_4h >= BTC_RECOVER_RSI)
        meta["method"] = "ema200+rsi"
        meta["btc_ema200_4h"] = m.btc_ema200_4h
    meta["btc_price"] = m.btc_price
    meta["btc_rsi_4h"] = m.btc_rsi_4h
    meta["btc_recover"] = recover
    return recover, meta


def decide(m: MarketSnapshot, p: PositionSnapshot, last_state: str) -> Decision:
    loan_active = p.loan_usdt > 0.0
    breakdown, bmeta = is_btc_breakdown(m)
    recover, rmeta = is_btc_recover(m)

    # Base state by LTV first
    if p.ltv_pct >= CRITICAL_LTV:
        state = "CRITICAL"
    elif p.ltv_pct >= RISK_LTV:
        state = "RISK"
    else:
        state = "SAFE" if p.ltv_pct <= SAFE_LTV else "RISK"

    # If BTC breakdown while loan active -> at least RISK
    if loan_active and breakdown:
        state = "CRITICAL" if p.ltv_pct >= (RISK_LTV) else "RISK"

    # If BTC recovers and LTV is low -> SAFE
    if loan_active and recover and p.ltv_pct <= SAFE_LTV:
        state = "SAFE"

    # Decision/action per state (the bot is NOT executing trades; only signals)
    meta = {
        "ltv_pct": round(p.ltv_pct, 2),
        "loan_active": loan_active,
        "loan_usdt": round(p.loan_usdt, 2),
        "collateral_usdt": round(p.collateral_usdt, 2),
        **bmeta,
    }

    # CRITICAL
    if state == "CRITICAL":
        return Decision(
            state=state,
            action="REDUCE_RISK",
            confidence="high",
            message="CRITICAL: Đang vay + BTC breakdown hoặc LTV cao → ưu tiên GIẢM NỢ/GIẢM RỦI RO, không tăng vị thế.",
            meta=meta,
        )

    # RISK
    if state == "RISK":
        # If oversold ETH but BTC breakdown: still not add
        if loan_active and breakdown:
            return Decision(
                state=state,
                action="REDUCE_RISK",
                confidence="high",
                message="RISK: BTC breakdown trong khi đang vay → ưu tiên giảm rủi ro/giảm nợ, KHÔNG tăng vị thế.",
                meta=meta,
            )

        # If BTC not breakdown but ETH oversold: allow "watch/buyback small" (not borrow-more)
        if (not breakdown) and (m.eth_rsi_4h <= ETH_OVERSOLD_RSI):
            return Decision(
                state=state,
                action="BUYBACK",
                confidence="med",
                message="RISK: ETH oversold (RSI thấp) và BTC chưa breakdown → có thể BUYBACK nhẹ (ưu tiên dùng vốn sẵn có, không tăng vay).",
                meta={**meta, "eth_rsi_4h": m.eth_rsi_4h, **rmeta},
            )

        return Decision(
            state=state,
            action="HOLD",
            confidence="med",
            message="RISK: Giữ nguyên. Không tăng vay. Chờ BTC hồi cấu trúc (reclaim EMA/RSI) rồi mới tính.",
            meta={**meta, **rmeta},
        )

    # SAFE
    return Decision(
        state=state,
        action="SAFE_HOLD",
        confidence="med",
        message="SAFE: LTV thấp và rủi ro BTC không xấu → có thể giữ/quan sát. Nếu muốn vay thêm thì chỉ tăng từng phần nhỏ + đặt giới hạn LTV.",
        meta={**meta, **rmeta},
    )


# =========================
# Anti-spam rules
# =========================

def should_notify(db: Session, decision: Decision) -> Tuple[bool, str]:
    """
    Notify only when:
      - state changes, OR
      - action changes, OR
      - cooldown passed for the state, OR
      - action min-interval passed for same action
    """
    last_state = db_get_state(db, "last_state", "UNKNOWN")
    last_action = db_get_state(db, "last_action", "NONE")

    last_notify_ts = float(db_get_state(db, "last_notify_ts", "0"))
    last_action_ts = float(db_get_state(db, "last_action_ts", "0"))

    now_ts = time.time()

    state_changed = (decision.state != last_state)
    action_changed = (decision.action != last_action)

    # strong rule: on state change, always notify
    if state_changed:
        return True, "state_changed"

    # if action changed within same state, notify
    if action_changed:
        # but avoid rapid flip-flop: respect short min interval
        if now_ts - last_action_ts >= 60:
            return True, "action_changed"
        return False, "action_flip_flop_guard"

    # same state & same action: respect cooldown
    if now_ts - last_notify_ts >= STATE_NOTIFY_COOLDOWN_SECONDS:
        return True, "state_cooldown"

    # extra guard: if same action persists, do not send too often
    if now_ts - last_action_ts >= ACTION_MIN_INTERVAL_SECONDS:
        return True, "action_min_interval"

    return False, "no_change"


def persist_decision(db: Session, m: MarketSnapshot, decision: Decision):
    row = SignalRow(
        state=decision.state,
        action=decision.action,
        confidence=decision.confidence,
        message=decision.message,
        meta_json=json.dumps(decision.meta, ensure_ascii=False),
        eth_price=m.eth_price,
        btc_price=m.btc_price,
        ethbtc_price=m.ethbtc_price,
    )
    db.add(row)
    db.commit()

    db_set_state(db, "last_state", decision.state)
    db_set_state(db, "last_action", decision.action)
    db_set_state(db, "last_action_ts", str(time.time()))
    db_set_state(db, "last_notify_ts", str(time.time()))


# =========================
# Bot runner
# =========================

async def bot_loop(app: FastAPI):
    # give app time to start
    await asyncio.sleep(2)
    while True:
        try:
            db = SessionLocal()
            last_state = db_get_state(db, "last_state", "UNKNOWN")

            m = fetch_market()
            p = fetch_position()
            d = decide(m, p, last_state)

            notify, reason = should_notify(db, d)

            # Always store the latest decision as a signal row ONLY if we notify
            # (prevents DB spam too). If you want every tick stored, change here.
            if notify:
                persist_decision(db, m, d)

                title = f"{d.action} | {d.state}"
                msg = (
                    f"{d.message}\n\n"
                    f"ETH: {m.eth_price:.2f} | BTC: {m.btc_price:.2f} | ETHBTC: {m.ethbtc_price:.5f}\n"
                    f"LTV: {p.ltv_pct:.2f}% | loan: {p.loan_usdt:.0f}\n"
                    f"Reason: {reason}"
                )
                pushover_send(title, msg, priority=0)

            else:
                # still update last_state/action in DB? NO.
                # Because no change => keep prior signal as "current".
                pass

        except Exception as e:
            # push error occasionally (also anti-spam)
            try:
                db = SessionLocal()
                last_err_ts = float(db_get_state(db, "last_err_ts", "0"))
                if time.time() - last_err_ts > 3600:
                    db_set_state(db, "last_err_ts", str(time.time()))
                    pushover_send("BOT ERROR", f"{type(e).__name__}: {str(e)[:800]}", priority=0)
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

        await asyncio.sleep(RUN_EVERY_SECONDS)


# =========================
# FastAPI + Dashboard
# =========================

app = FastAPI(title=APP_TITLE)


@app.on_event("startup")
async def on_startup():
    init_db()
    asyncio.create_task(bot_loop(app))


def render_dashboard(latest: Optional[SignalRow], signals: List[SignalRow], cfg: Dict[str, Any]) -> str:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    latest_block = ""
    if latest:
        latest_block = f"""
        <div class="card">
          <div class="h">Latest</div>
          <div><b>{esc(latest.action)}</b> | state: <b>{esc(latest.state)}</b> | {latest.created_at}</div>
          <div style="margin-top:8px">{esc(latest.message)}</div>
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

    cfg_pre = esc(json.dumps(cfg, indent=2, ensure_ascii=False))

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>{APP_TITLE}</title>
      <style>
        body {{ font-family: ui-sans-serif, system-ui; background:#0b0f15; color:#e6edf3; padding:24px; }}
        .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
        .card {{ background:#111827; border:1px solid #1f2937; border-radius:16px; padding:16px; min-width:320px; flex:1; }}
        .h {{ font-size:18px; font-weight:700; margin-bottom:10px; }}
        table {{ width:100%; border-collapse:collapse; margin-top:16px; }}
        th, td {{ border-bottom:1px solid #1f2937; padding:10px; font-size:13px; vertical-align:top; }}
        th {{ text-align:left; color:#9ca3af; font-weight:600; }}
        .pre {{ background:#0b1220; border:1px solid #1f2937; border-radius:12px; padding:12px; overflow:auto; }}
        a {{ color:#60a5fa; }}
      </style>
    </head>
    <body>
      <div class="row">
        <div class="card">
          <div class="h">Config</div>
          <div>RUN_EVERY_SECONDS: <b>{RUN_EVERY_SECONDS}</b> (10 phút = 600)</div>
          <div>STATE_NOTIFY_COOLDOWN_SECONDS: <b>{STATE_NOTIFY_COOLDOWN_SECONDS}</b></div>
          <div>ACTION_MIN_INTERVAL_SECONDS: <b>{ACTION_MIN_INTERVAL_SECONDS}</b></div>
          <div style="margin-top:10px">Market endpoint: <code>{esc(MARKET_ENDPOINT)}</code></div>
          <div>Position endpoint: <code>{esc(POSITION_ENDPOINT)}</code></div>
          <pre class="pre">{cfg_pre}</pre>
          <div style="margin-top:10px">
            <a href="/api/latest">/api/latest</a> ·
            <a href="/api/signals?limit=50">/api/signals</a> ·
            <a href="/api/state">/api/state</a>
          </div>
        </div>
        {latest_block}
      </div>

      <div class="card" style="margin-top:16px">
        <div class="h">Recent Signals</div>
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Action</th><th>State</th><th>Conf</th><th>Message</th><th>ETH</th><th>BTC</th><th>ETHBTC</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home(db: Session = Depends(get_db)):
    latest = db.query(SignalRow).order_by(SignalRow.created_at.desc()).first()
    signals = db.query(SignalRow).order_by(SignalRow.created_at.desc()).limit(50).all()

    cfg = {
        "SAFE_LTV": SAFE_LTV,
        "RISK_LTV": RISK_LTV,
        "CRITICAL_LTV": CRITICAL_LTV,
        "BTC_BREAKDOWN_RSI": BTC_BREAKDOWN_RSI,
        "BTC_RECOVER_RSI": BTC_RECOVER_RSI,
        "ETH_OVERSOLD_RSI": ETH_OVERSOLD_RSI,
    }
    return HTMLResponse(render_dashboard(latest, signals, cfg))


@app.get("/api/state")
def api_state(db: Session = Depends(get_db)):
    return {
        "last_state": db_get_state(db, "last_state", "UNKNOWN"),
        "last_action": db_get_state(db, "last_action", "NONE"),
        "last_notify_ts": db_get_state(db, "last_notify_ts", "0"),
        "last_action_ts": db_get_state(db, "last_action_ts", "0"),
        "run_every_seconds": RUN_EVERY_SECONDS,
        "cooldown_seconds": STATE_NOTIFY_COOLDOWN_SECONDS,
    }


@app.get("/api/latest")
def api_latest(db: Session = Depends(get_db)):
    s = db.query(SignalRow).order_by(SignalRow.created_at.desc()).first()
    if not s:
        return {"ok": True, "latest": None}
    return {
        "ok": True,
        "latest": {
            "time": s.created_at.isoformat(),
            "state": s.state,
            "action": s.action,
            "confidence": s.confidence,
            "message": s.message,
            "meta": json.loads(s.meta_json),
            "eth_price": s.eth_price,
            "btc_price": s.btc_price,
            "ethbtc_price": s.ethbtc_price,
        }
    }


@app.get("/api/signals")
def api_signals(limit: int = 100, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 500))
    rows = db.query(SignalRow).order_by(SignalRow.created_at.desc()).limit(limit).all()
    return {
        "ok": True,
        "signals": [
            {
                "time": r.created_at.isoformat(),
                "state": r.state,
                "action": r.action,
                "confidence": r.confidence,
                "message": r.message,
                "meta": json.loads(r.meta_json),
                "eth_price": r.eth_price,
                "btc_price": r.btc_price,
                "ethbtc_price": r.ethbtc_price,
            }
            for r in rows
        ]
    }


# =========================
# Backtest
# =========================

class BacktestItem(BaseModel):
    # minimal fields you already have in logs
    time: str
    eth_price: float
    btc_price: float
    ethbtc_price: float
    eth_rsi_4h: float
    btc_rsi_4h: float
    ethbtc_rsi_4h: float
    eth_bb_lower_4h: Optional[float] = None
    eth_bb_mid_4h: Optional[float] = None
    eth_bb_upper_4h: Optional[float] = None
    btc_ema200_4h: Optional[float] = None
    loan_usdt: float = 0.0
    collateral_usdt: float = 1.0
    ltv_pct: float = 0.0
    spot_eth: float = 0.0
    avg_entry: float = 0.0


@app.post("/api/backtest")
def api_backtest(items: List[BacktestItem]):
    """
    Feed array of snapshots, it returns only "events" (state/action changes)
    This mirrors anti-spam behavior.
    """
    if not items:
        return {"ok": True, "events": []}

    last_state = "UNKNOWN"
    last_action = "NONE"
    last_notify_ts = 0.0
    last_action_ts = 0.0

    events = []

    for it in items:
        m = MarketSnapshot(
            time=dt.datetime.fromisoformat(it.time.replace("Z", "+00:00")).replace(tzinfo=None),
            eth_price=it.eth_price,
            btc_price=it.btc_price,
            ethbtc_price=it.ethbtc_price,
            eth_rsi_4h=it.eth_rsi_4h,
            btc_rsi_4h=it.btc_rsi_4h,
            ethbtc_rsi_4h=it.ethbtc_rsi_4h,
            eth_bb_lower_4h=it.eth_bb_lower_4h,
            eth_bb_mid_4h=it.eth_bb_mid_4h,
            eth_bb_upper_4h=it.eth_bb_upper_4h,
            btc_ema200_4h=it.btc_ema200_4h,
        )
        p = PositionSnapshot(
            time=m.time,
            loan_usdt=it.loan_usdt,
            collateral_usdt=max(it.collateral_usdt, 1.0),
            ltv_pct=it.ltv_pct,
            spot_eth=it.spot_eth,
            avg_entry=it.avg_entry,
        )

        d = decide(m, p, last_state)

        now_ts = m.time.timestamp()
        state_changed = d.state != last_state
        action_changed = d.action != last_action

        notify = False
        reason = "no_change"
        if state_changed:
            notify = True
            reason = "state_changed"
        elif action_changed and (now_ts - last_action_ts >= 60):
            notify = True
            reason = "action_changed"
        elif (now_ts - last_notify_ts) >= STATE_NOTIFY_COOLDOWN_SECONDS:
            notify = True
            reason = "state_cooldown"
        elif (now_ts - last_action_ts) >= ACTION_MIN_INTERVAL_SECONDS:
            notify = True
            reason = "action_min_interval"

        if notify:
            events.append({
                "time": it.time,
                "state": d.state,
                "action": d.action,
                "confidence": d.confidence,
                "message": d.message,
                "meta": d.meta,
                "reason": reason,
                "eth_price": it.eth_price,
                "btc_price": it.btc_price,
                "ethbtc_price": it.ethbtc_price,
                "ltv_pct": it.ltv_pct,
            })
            last_notify_ts = now_ts

        last_state = d.state
        last_action = d.action
        last_action_ts = now_ts

    return {"ok": True, "events": events}
