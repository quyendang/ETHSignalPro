import os
import time
import threading
from dataclasses import dataclass, asdict
from math import isnan
from typing import List, Dict, Any, Optional
import math  # thêm dòng này
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

#############################
# ENV & GLOBAL CONFIG
#############################

INITIAL_SPOT_ETH = float(os.getenv("INITIAL_SPOT_ETH", "138"))
AVG_ENTRY_ETH = float(os.getenv("AVG_ENTRY_ETH", "3150"))

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "eth_signals")

TRAILING_PCT = float(os.getenv("TRAILING_PCT", "0.03"))  # 3%

# In-memory state for dashboard
RECENT_SIGNALS: List[Dict[str, Any]] = []
MAX_SIGNALS = 100

LAST_MARKET_STATE: Dict[str, Any] = {}
LAST_CONFIG: Dict[str, Any] = {}

#############################
# Helpers: Notification
#############################

def pushover_notify(title: str, message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        print("[PUSHOVER] Missing credentials")
        return
    data = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": title,
        "message": message,
        "priority": 0,
        "sound": "belll",
    }
    if PUSHOVER_DEVICE:
        data["device"] = PUSHOVER_DEVICE
    try:
        requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=10)
    except Exception as e:
        print("[PUSHOVER ERROR]", e)


def telegram_notify(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Missing credentials")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("[TELEGRAM ERROR]", e)


def supabase_log_signal(action: str, m: "MarketState", payload: Dict[str, Any]):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[SUPABASE] Missing config")
        return
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    data = {
        "action": action,
        "eth_price": m.eth_price,
        "btc_price": m.btc_price,
        "ethbtc_price": m.ethbtc_price,
        "payload": payload,
    }
    try:
        requests.post(url, json=data, headers=headers, timeout=10)
    except Exception as e:
        print("[SUPABASE ERROR]", e)


#############################
# Indicators
#############################

def calc_rsi(closes: List[float], period=14) -> float:
    if len(closes) < period + 5:
        return float("nan")
    arr = np.array(closes, dtype=float)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    if avg_loss == 0:
        return 100.0
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100/(1+rs))


def calc_ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) < period:
        return np.full_like(series, np.nan)
    ema = np.zeros_like(series)
    k = 2 / (period + 1)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = series[i]*k + ema[i-1]*(1-k)
    return ema


def calc_macd(closes: List[float], fast=12, slow=26, signal=9):
    if len(closes) < slow + signal + 5:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(closes, dtype=float)
    ema_fast = calc_ema(arr, fast)
    ema_slow = calc_ema(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(hist[-1])


def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 5:
        return float("nan")
    trs = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)
    arr = np.array(trs, dtype=float)
    return float(arr[-period:].mean())


def realized_vol(closes: List[float], lookback=60):
    if len(closes) < lookback + 2:
        return float("nan")
    arr = np.array(closes[-lookback:], dtype=float)
    rets = np.diff(np.log(arr))
    return float(np.std(rets))


def qtile(values: List[float], q: float):
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    return float(np.quantile(arr, q))


def bollinger(closes: List[float], period: int = 20, k: float = 2.0):
    arr = np.array(closes, dtype=float)
    if len(arr) < period:
        return float("nan"), float("nan"), float("nan")
    window = arr[-period:]
    ma = window.mean()
    std = window.std()
    upper = ma + k * std
    lower = ma - k * std
    return float(lower), float(ma), float(upper)


#############################
# Bybit Client (public REST v5)
#############################

class BybitClient:
    BASE_URL = "https://api.bybit.com"

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL + path
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                print("[BYBIT ERROR]", data.get("retMsg"), data)
                return None
            return data
        except Exception as e:
            print("[BYBIT HTTP ERROR]", e)
            return None

    def get_ticker(self, symbol: str) -> Dict[str, float]:
        data = self._get("/v5/market/tickers", {
            "category": "spot",
            "symbol": symbol,
        })
        if not data or "result" not in data or not data["result"].get("list"):
            return {"last": float("nan")}
        last_str = data["result"]["list"][0]["lastPrice"]
        return {"last": float(last_str)}

    def _map_interval(self, interval: str) -> str:
        mapping = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1D": "D",
        }
        return mapping.get(interval, interval)

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 200) -> List[Dict[str, Any]]:
        bybit_interval = self._map_interval(interval)
        limit = min(limit, 200)
        data = self._get("/v5/market/kline", {
            "category": "spot",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit,
        })
        if not data or "result" not in data or not data["result"].get("list"):
            return []
        candles = []
        for item in reversed(data["result"]["list"]):
            candles.append({
                "timestamp": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            })
        return candles

    def get_balance(self, asset: str) -> float:
        if asset == "ETH":
            return INITIAL_SPOT_ETH
        return 0.0


#############################
# Data structures
#############################

@dataclass
class MarketState:
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    btc_rsi_4h: float

    eth_macd_hist_4h: float
    btc_macd_hist_4h: float

    ethbtc_rsi_4h: float
    ethbtc_rsi_1d: float
    ethbtc_trend: str  # bull / bear / neutral


@dataclass
class PositionState:
    spot_eth: float
    loan_usdt: float
    avg_entry_eth: float
    last_sell_price: Optional[float] = None
    last_sell_size: Optional[float] = None


#############################
# History Provider
#############################

class HistoryProvider:
    def __init__(self, client: BybitClient):
        self.client = client

    def closes(self, symbol: str, interval: str, limit: int) -> List[float]:
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["close"] for c in candles]

    def highs(self, symbol: str, interval: str, limit: int) -> List[float]:
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["high"] for c in candles]

    def lows(self, symbol: str, interval: str, limit: int) -> List[float]:
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["low"] for c in candles]

    def rsi_series(self, symbol: str, interval: str, lookback: int, period=14) -> List[float]:
        closes = self.closes(symbol, interval, lookback + period + 5)
        out = []
        for i in range(period + 1, len(closes)):
            out.append(calc_rsi(closes[:i], period=period))
        return out


#############################
# Dynamic Config Builder
#############################

class DynamicConfigBuilder:
    def __init__(self, hist: HistoryProvider):
        self.h = hist

    def build(self) -> Dict[str, Any]:
        # RSI dynamic ETH/BTC (4H)
        eth_rsi_4h_series = self.h.rsi_series("ETHUSDT", "4h", 200)
        btc_rsi_4h_series = self.h.rsi_series("BTCUSDT", "4h", 200)

        eth_rsi_overbought = qtile(eth_rsi_4h_series, 0.8)
        eth_rsi_oversold = qtile(eth_rsi_4h_series, 0.2)
        btc_rsi_overbought = qtile(btc_rsi_4h_series, 0.8)
        btc_rsi_oversold = qtile(btc_rsi_4h_series, 0.2)

        # BTC dump level (MA20 - 1.5*ATR) trên 1D
        daily_btc = self.h.client.get_ohlcv("BTCUSDT", "1d", 60)
        if not daily_btc:
            btc_dump_level = 90000.0
        else:
            closes = [c["close"] for c in daily_btc]
            highs = [c["high"] for c in daily_btc]
            lows = [c["low"] for c in daily_btc]
            if len(closes) < 20:
                btc_dump_level = closes[-1] * 0.92
            else:
                ma20 = sum(closes[-20:]) / 20
                atr14 = calc_atr(highs, lows, closes)
                btc_dump_level = ma20 - 1.5 * atr14

        # Buyback zone + Bollinger ETH trên 4H
        eth_4h = self.h.client.get_ohlcv("ETHUSDT", "4h", 120)
        closes_4h = [c["close"] for c in eth_4h] if eth_4h else []

        if len(closes_4h) < 30:
            buyback_low, buyback_high = 0.0, 0.0
            sell_spot_eth_price_min = 0.0
            bb_lower, bb_mid, bb_upper = float("nan"), float("nan"), float("nan")
        else:
            recent = closes_4h[-60:]
            local_low = min(recent)
            local_high = max(recent)
            fib50 = local_low + 0.5 * (local_high - local_low)
            fib618 = local_low + 0.618 * (local_high - local_low)
            buyback_low = (local_low + fib50) / 2
            buyback_high = fib618
            sell_spot_eth_price_min = local_high - 0.2 * (local_high - local_low)

            bb_lower, bb_mid, bb_upper = bollinger(closes_4h, period=20, k=2.0)

        # Volatility → max LTV
        vol = realized_vol(closes_4h) if closes_4h else float("nan")
        if isnan(vol):
            max_ltv = 0.3
        elif vol < 0.02:
            max_ltv = 0.4
        elif vol < 0.04:
            max_ltv = 0.3
        else:
            max_ltv = 0.2

        # ETHBTC breakdown & regime (4H + 1D)
        ethbtc_4h = self.h.client.get_ohlcv("ETHBTC", "4h", 200)
        ethbtc_closes = [c["close"] for c in ethbtc_4h] if ethbtc_4h else []
        ethbtc_highs = [c["high"] for c in ethbtc_4h] if ethbtc_4h else []
        ethbtc_lows = [c["low"] for c in ethbtc_4h] if ethbtc_4h else []

        if len(ethbtc_closes) >= 60:
            ma50 = sum(ethbtc_closes[-50:]) / 50
            atr_ethbtc = calc_atr(ethbtc_highs, ethbtc_lows, ethbtc_closes)
            ethbtc_breakdown = ma50 - 0.5 * atr_ethbtc
        elif ethbtc_closes:
            ethbtc_breakdown = ethbtc_closes[-1] * 0.97
        else:
            ethbtc_breakdown = 0.0

        # ETHBTC regime từ 1D
        ethbtc_1d = self.h.client.get_ohlcv("ETHBTC", "1d", 220)
        ethbtc_daily_closes = [c["close"] for c in ethbtc_1d] if ethbtc_1d else []
        if ethbtc_daily_closes:
            ethbtc_rsi_1d = calc_rsi(ethbtc_daily_closes)
            if len(ethbtc_daily_closes) >= 200:
                ma200 = sum(ethbtc_daily_closes[-200:]) / 200
            else:
                ma200 = ethbtc_daily_closes[-1]
            last = ethbtc_daily_closes[-1]
            if last > ma200 and ethbtc_rsi_1d > 50:
                ethbtc_regime = "bull"
            elif last < ma200 and ethbtc_rsi_1d < 50:
                ethbtc_regime = "bear"
            else:
                ethbtc_regime = "neutral"
        else:
            ethbtc_rsi_1d = float("nan")
            ethbtc_regime = "neutral"

        # MA200 1D cho ETH
        eth_1d = self.h.client.get_ohlcv("ETHUSDT", "1d", 220)
        eth_1d_closes = [c["close"] for c in eth_1d] if eth_1d else []
        if len(eth_1d_closes) >= 200:
            eth_ma200_1d = sum(eth_1d_closes[-200:]) / 200
        elif eth_1d_closes:
            eth_ma200_1d = eth_1d_closes[-1]
        else:
            eth_ma200_1d = 0.0

        conf = {
            "eth_rsi_overbought_4h": eth_rsi_overbought,
            "eth_rsi_oversold_4h": eth_rsi_oversold,
            "btc_rsi_overbought_4h": btc_rsi_overbought,
            "btc_rsi_oversold_4h": btc_rsi_oversold,
            "btc_dump_level": btc_dump_level,
            "buyback_zone": (buyback_low, buyback_high),
            "max_ltv": max_ltv,
            "ethbtc_breakdown": ethbtc_breakdown,
            "sell_spot_eth_price_min": sell_spot_eth_price_min,
            "ethbtc_regime": ethbtc_regime,
            "ethbtc_rsi_1d": ethbtc_rsi_1d,
            "eth_bb_lower_4h": bb_lower,
            "eth_bb_mid_4h": bb_mid,
            "eth_bb_upper_4h": bb_upper,
            "eth_ma200_1d": eth_ma200_1d,
        }
        return conf


#############################
# Strategy (with trailing + cooldown)
#############################

class EthStrategy:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.prev_eth_macd_hist_4h: Optional[float] = None

        # trailing state
        self.trailing_active = False
        self.trailing_peak = None

        # SELL cooldown
        self.cooldown_secs = 4 * 60 * 60  # 4h
        self.last_sell_ts: float = 0.0

    def update_conf(self, conf: Dict[str, Any]):
        self.conf = conf

    def _update_trailing(self, price: float) -> Optional[Dict[str, Any]]:
        if not self.trailing_active or self.trailing_peak is None:
            return None
        if price > self.trailing_peak:
            self.trailing_peak = price
            return None
        trigger_price = self.trailing_peak * (1 - TRAILING_PCT)
        if price <= trigger_price:
            # trailing stop hit → tắt trailing, trả info cho caller
            self.trailing_active = False
            peak = self.trailing_peak
            self.trailing_peak = None
            return {
                "triggered": True,
                "trigger_price": trigger_price,
                "peak_price": peak,
            }
        return None

    def _maybe_start_trailing(self, price: float):
        if not self.trailing_active:
            self.trailing_active = True
            self.trailing_peak = price

    def should_sell_spot(self, m: "MarketState", p: "PositionState") -> Optional[Dict]:
        # 1. Nếu trailing đang active → check trigger
        trailing = self._update_trailing(m.eth_price)
        if trailing and trailing.get("triggered"):
            # Check cooldown SELL 4h
            now = time.time()
            if now - self.last_sell_ts < self.cooldown_secs:
                # Trong thời gian cooldown → bỏ qua SELL lần này
                return None

            size = min(30.0, p.spot_eth * 0.3)
            if size <= 0:
                return None
            return {
                "action": "SELL_SPOT_ETH",
                "size": round(size, 4),
                "mode": "trailing",
                "reason": {
                    "eth_price": m.eth_price,
                    "trailing_info": trailing,
                },
            }

        # 2. Nếu chưa trailing, check điều kiện để bật trailing
        min_price = self.conf.get("sell_spot_eth_price_min", 0.0)
        if m.eth_price < min_price or min_price <= 0:
            self.prev_eth_macd_hist_4h = m.eth_macd_hist_4h
            return None

        # Filter: ETHBTC bull regime → hạn chế swing SELL
        if self.conf.get("ethbtc_regime") == "bull":
            self.prev_eth_macd_hist_4h = m.eth_macd_hist_4h
            return None

        # Điều kiện Bollinger: ưu tiên chỉ bật trailing khi giá chạm near upper band
        bb_upper = self.conf.get("eth_bb_upper_4h", float("nan"))
        if not isnan(bb_upper) and m.eth_price < bb_upper:
            self.prev_eth_macd_hist_4h = m.eth_macd_hist_4h
            return None

        rsi_hot = (
            not isnan(m.eth_rsi_4h)
            and not isnan(self.conf.get("eth_rsi_overbought_4h", float("nan")))
            and m.eth_rsi_4h >= self.conf["eth_rsi_overbought_4h"]
        )

        macd_peak = False
        if self.prev_eth_macd_hist_4h is not None and not isnan(m.eth_macd_hist_4h):
            macd_peak = (
                self.prev_eth_macd_hist_4h > 0
                and m.eth_macd_hist_4h > 0
                and m.eth_macd_hist_4h < self.prev_eth_macd_hist_4h
            )

        bad_ethbtc = (
            self.conf.get("ethbtc_breakdown", 0) > 0
            and m.ethbtc_price < self.conf["ethbtc_breakdown"]
        )
        bad_btc = m.btc_price < self.conf["btc_dump_level"]

        self.prev_eth_macd_hist_4h = m.eth_macd_hist_4h

        if (rsi_hot or macd_peak) and (bad_ethbtc or bad_btc):
            # Điều kiện đủ kém → bật trailing (chưa SELL ngay)
            self._maybe_start_trailing(m.eth_price)
            return None

        return None

    def should_buyback_spot(self, m: "MarketState", p: "PositionState") -> Optional[Dict]:
        if not p.last_sell_size or p.last_sell_size <= 0:
            return None

        low, high = self.conf.get("buyback_zone", (0, 0))
        if low <= 0 or high <= 0:
            return None

        if not (low <= m.eth_price <= high):
            return None

        if m.btc_price < self.conf["btc_dump_level"] * 0.95:
            return None

        if not isnan(self.conf.get("eth_rsi_overbought_4h", float("nan"))):
            if m.eth_rsi_4h > self.conf["eth_rsi_overbought_4h"]:
                return None

        # Ưu tiên buyback gần lower BB
        bb_lower = self.conf.get("eth_bb_lower_4h", float("nan"))
        if not isnan(bb_lower) and m.eth_price > bb_lower:
            # giá chưa về tới lower band => có thể bỏ qua buyback lần này
            return None

        return {
            "action": "BUYBACK_SPOT_ETH",
            "size": round(p.last_sell_size, 4),
            "reason": {
                "eth_price": m.eth_price,
                "buyback_zone": (low, high),
                "btc_price": m.btc_price,
                "eth_rsi_4h": m.eth_rsi_4h,
                "bb_lower_4h": bb_lower,
            },
        }

    def max_safe_loan(self, m: "MarketState", p: "PositionState") -> float:
        if p.spot_eth <= 0:
            return 0.0
        collateral_value = p.spot_eth * m.eth_price
        if collateral_value <= 0:
            return 0.0
        max_ltv = self.conf.get("max_ltv", 0.3)
        current_ltv = p.loan_usdt / collateral_value if p.loan_usdt > 0 else 0.0
        room = max_ltv - current_ltv
        if room <= 0:
            return 0.0
        return collateral_value * room

    def should_open_loan(self, m: "MarketState", p: "PositionState") -> Optional[Dict]:
        low, high = self.conf.get("buyback_zone", (0, 0))
        if low <= 0 or high <= 0:
            return None

        # Filter: chỉ vay khi ETH trên MA200 1D
        ma200 = self.conf.get("eth_ma200_1d", 0.0)
        if ma200 > 0 and m.eth_price < ma200:
            return None

        # Chỉ vay-mua thêm khi ETH đang trong/ dưới vùng chiết khấu
        if m.eth_price > high:
            return None

        if m.btc_price < self.conf["btc_dump_level"]:
            return None

        if self.conf.get("ethbtc_breakdown", 0) > 0 and m.ethbtc_price < self.conf["ethbtc_breakdown"]:
            return None

        loan_amount = self.max_safe_loan(m, p)
        if loan_amount <= 0:
            return None

        loan_amount = min(loan_amount, 50_000.0)

        return {
            "action": "OPEN_LOAN_BUY_ETH",
            "loan_amount": round(loan_amount, 2),
            "reason": {
                "eth_price": m.eth_price,
                "btc_price": m.btc_price,
                "ethbtc_price": m.ethbtc_price,
                "max_ltv": self.conf.get("max_ltv", 0.3),
                "eth_ma200_1d": ma200,
            },
        }

    def should_repay_loan(self, m: "MarketState", p: "PositionState") -> Optional[Dict]:
        if p.loan_usdt <= 0:
            return None

        low, high = self.conf.get("buyback_zone", (0, 0))
        if high <= 0:
            return None
        target_price = high * 1.10
        if m.eth_price < target_price:
            return None

        if not isnan(self.conf.get("eth_rsi_overbought_4h", float("nan"))):
            if m.eth_rsi_4h < self.conf["eth_rsi_overbought_4h"]:
                return None

        buffer = 1.01
        size_to_sell = (p.loan_usdt * buffer) / m.eth_price
        if size_to_sell <= 0:
            return None

        return {
            "action": "REPAY_LOAN_SELL_ETH",
            "size": round(size_to_sell, 4),
            "reason": {
                "eth_price": m.eth_price,
                "loan_usdt": p.loan_usdt,
                "target_price": target_price,
            },
        }


#############################
# ETH Bot
#############################

class EthBot:
    def __init__(self):
        self.client = BybitClient()
        self.hist = HistoryProvider(self.client)
        self.conf_builder = DynamicConfigBuilder(self.hist)
        self.conf = self.conf_builder.build()
        self.strategy = EthStrategy(self.conf)

        self.position = PositionState(
            spot_eth=INITIAL_SPOT_ETH,
            loan_usdt=0.0,
            avg_entry_eth=AVG_ENTRY_ETH,
        )

        self.last_conf_update = 0
        self.conf_update_interval = 4 * 60 * 60  # 4h

    def update_position(self):
        # nếu sau này bạn muốn đọc balance thật từ ENV khác thì chỉnh lại chỗ này
        self.position.spot_eth = INITIAL_SPOT_ETH

    def build_market_state(self) -> MarketState:
        eth = self.client.get_ticker("ETHUSDT")["last"]
        btc = self.client.get_ticker("BTCUSDT")["last"]
        ethbtc = self.client.get_ticker("ETHBTC")["last"]

        eth4 = self.hist.closes("ETHUSDT", "4h", 80)
        btc4 = self.hist.closes("BTCUSDT", "4h", 80)
        ethbtc4 = self.hist.closes("ETHBTC", "4h", 80)
        ethbtc1d = self.hist.closes("ETHBTC", "1d", 120)

        eth_rsi_4h = calc_rsi(eth4) if eth4 else float("nan")
        btc_rsi_4h = calc_rsi(btc4) if btc4 else float("nan")

        _, _, eth_macd_hist_4h = calc_macd(eth4) if eth4 else (float("nan"),)*3
        _, _, btc_macd_hist_4h = calc_macd(btc4) if btc4 else (float("nan"),)*3

        ethbtc_rsi_4h = calc_rsi(ethbtc4) if ethbtc4 else float("nan")
        ethbtc_rsi_1d = calc_rsi(ethbtc1d) if ethbtc1d else float("nan")
        ethbtc_trend = self.conf.get("ethbtc_regime", "neutral")

        m = MarketState(
            eth_price=eth,
            btc_price=btc,
            ethbtc_price=ethbtc,
            eth_rsi_4h=eth_rsi_4h,
            btc_rsi_4h=btc_rsi_4h,
            eth_macd_hist_4h=eth_macd_hist_4h,
            btc_macd_hist_4h=btc_macd_hist_4h,
            ethbtc_rsi_4h=ethbtc_rsi_4h,
            ethbtc_rsi_1d=ethbtc_rsi_1d,
            ethbtc_trend=ethbtc_trend,
        )
        return m

    def maybe_update_conf(self):
        global LAST_CONFIG
        now = time.time()
        if now - self.last_conf_update > self.conf_update_interval:
            self.conf = self.conf_builder.build()
            self.strategy.update_conf(self.conf)
            self.last_conf_update = now
            LAST_CONFIG = self.conf.copy()
            msg = f"Dynamic config updated: {self.conf}"
            print("[CONF]", msg)
            pushover_notify("ETH Bot – Config Updated", msg)
            telegram_notify(msg)

    def handle_signal(self, signal: Dict[str, Any], m: MarketState):
        global RECENT_SIGNALS

        action = signal["action"]
        payload = {
            "action": action,
            "signal": signal,
            "market": asdict(m),
            "ts": int(time.time()),
        }
        print("[SIGNAL]", payload)

        RECENT_SIGNALS.append(payload)
        if len(RECENT_SIGNALS) > MAX_SIGNALS:
            RECENT_SIGNALS = RECENT_SIGNALS[-MAX_SIGNALS:]

        title = f"ETH SIGNAL: {action}"
        msg = (
            f"{action}\n"
            f"ETH: {m.eth_price:.2f}\n"
            f"BTC: {m.btc_price:.2f}\n"
            f"ETHBTC: {m.ethbtc_price:.6f}\n\n"
            f"{signal}"
        )
        pushover_notify(title, msg)
        telegram_notify(msg)

        supabase_log_signal(action, m, payload)

        # Update “virtual” position
        if action == "SELL_SPOT_ETH":
            size = signal["size"]
            self.position.spot_eth = max(0.0, self.position.spot_eth - size)
            self.position.last_sell_price = m.eth_price
            self.position.last_sell_size = size
            # ghi nhận thời điểm SELL cho cooldown
            self.strategy.last_sell_ts = time.time()

        elif action == "BUYBACK_SPOT_ETH":
            size = signal["size"]
            self.position.spot_eth += size
            self.position.last_sell_price = None
            self.position.last_sell_size = None

        elif action == "OPEN_LOAN_BUY_ETH":
            loan = signal["loan_amount"]
            self.position.loan_usdt += loan
            bought_eth = loan / m.eth_price
            self.position.spot_eth += bought_eth

        elif action == "REPAY_LOAN_SELL_ETH":
            size = signal["size"]
            self.position.spot_eth = max(0.0, self.position.spot_eth - size)
            self.position.loan_usdt = 0.0

    def run_step(self):
        global LAST_MARKET_STATE

        self.maybe_update_conf()
        self.update_position()
        m = self.build_market_state()
        LAST_MARKET_STATE = asdict(m)

        sig = self.strategy.should_repay_loan(m, self.position)
        if not sig:
            sig = self.strategy.should_sell_spot(m, self.position)
        if not sig:
            sig = self.strategy.should_buyback_spot(m, self.position)
        if not sig:
            sig = self.strategy.should_open_loan(m, self.position)

        if sig:
            self.handle_signal(sig, m)


#############################
# FastAPI App + Bot loop
#############################

def sanitize_for_json(obj):
    """
    Đệ quy: thay NaN / +inf / -inf bằng None để JSONResponse không lỗi.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    # các kiểu khác (str, int, bool, None, ...) giữ nguyên
    return obj




app = FastAPI(title="ETH Strategy Bot – Bybit (BB + MA200 + Cooldown)")

bot = EthBot()

def bot_loop():
    while True:
        try:
            bot.run_step()
        except Exception as e:
            print("[BOT ERROR]", e)
            pushover_notify("ETH BOT ERROR", str(e))
        time.sleep(60)  # mỗi phút chạy 1 lần


@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("[BOT] Started background loop")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    m = LAST_MARKET_STATE or {}
    c = LAST_CONFIG or {}

    def fmt_float(val, digits=2):
        return f"{val:.{digits}f}" if isinstance(val, (int, float)) and not isnan(val) else "N/A"

    html_signals = ""
    for s in reversed(RECENT_SIGNALS[-20:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["ts"]))
        mk = s["market"]
        html_signals += f"""
        <tr>
          <td>{ts}</td>
          <td>{s['action']}</td>
          <td>{fmt_float(mk.get('eth_price'), 2)}</td>
          <td>{fmt_float(mk.get('btc_price'), 2)}</td>
          <td>{fmt_float(mk.get('ethbtc_price'), 6)}</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
      <title>ETH Strategy Bot Dashboard (Bybit)</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 20px;
          background-color: #0b1020;
          color: #f0f0f0;
        }}
        h1, h2 {{
          color: #ffd166;
        }}
        .card {{
          background: #141a33;
          border-radius: 10px;
          padding: 16px;
          margin-bottom: 20px;
          box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
        }}
        th, td {{
          border-bottom: 1px solid #333;
          padding: 6px 8px;
          font-size: 13px;
        }}
        th {{
          background: #20263f;
        }}
        pre {{
          white-space: pre-wrap;
          font-size: 12px;
        }}
      </style>
    </head>
    <body>
      <h1>ETH Strategy Bot (Bybit)</h1>

      <div class="card">
        <h2>Market State</h2>
        <p>ETH: {fmt_float(m.get('eth_price'), 2)} USDT</p>
        <p>BTC: {fmt_float(m.get('btc_price'), 2)} USDT</p>
        <p>ETHBTC: {fmt_float(m.get('ethbtc_price'), 6)}</p>
        <p>ETH RSI 4H: {fmt_float(m.get('eth_rsi_4h'), 2)}</p>
        <p>BTC RSI 4H: {fmt_float(m.get('btc_rsi_4h'), 2)}</p>
        <p>ETHBTC RSI 4H: {fmt_float(m.get('ethbtc_rsi_4h'), 2)}</p>
        <p>ETHBTC RSI 1D: {fmt_float(m.get('ethbtc_rsi_1d'), 2)}</p>
        <p>ETHBTC trend: {m.get('ethbtc_trend', 'N/A')}</p>
      </div>

      <div class="card">
        <h2>Dynamic Config</h2>
        <pre>{c}</pre>
      </div>

      <div class="card">
        <h2>Recent Signals</h2>
        <table>
          <tr>
            <th>Time</th>
            <th>Action</th>
            <th>ETH</th>
            <th>BTC</th>
            <th>ETHBTC</th>
          </tr>
          {html_signals if html_signals else "<tr><td colspan='5'>No signals yet</td></tr>"}
        </table>
      </div>
    </body>
    </html>
    """
    return html


@app.get("/api/state", response_class=JSONResponse)
def api_state():
    raw = {
        "market": LAST_MARKET_STATE,
        "config": LAST_CONFIG,
        "recent_signals": RECENT_SIGNALS[-50:],
    }
    safe = sanitize_for_json(raw)
    return safe

