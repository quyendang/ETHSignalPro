import os
import time
import threading
import json
from dataclasses import dataclass, asdict
from math import isnan
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
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
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=data, headers=headers, timeout=10)
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

    def get_ohlcv_range(self, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """Lấy dữ liệu OHLCV trong khoảng thời gian (timestamps in seconds)"""
        bybit_interval = self._map_interval(interval)
        all_candles = []
        
        # Bybit API uses milliseconds for start/end
        start_ms = start_ts * 1000
        end_ms = end_ts * 1000
        
        # Calculate interval in milliseconds
        interval_ms_map = {
            "1": 60 * 1000,
            "3": 3 * 60 * 1000,
            "5": 5 * 60 * 1000,
            "15": 15 * 60 * 1000,
            "30": 30 * 60 * 1000,
            "60": 60 * 60 * 1000,
            "120": 2 * 60 * 60 * 1000,
            "240": 4 * 60 * 60 * 1000,
            "360": 6 * 60 * 60 * 1000,
            "720": 12 * 60 * 60 * 1000,
            "D": 24 * 60 * 60 * 1000,
        }
        interval_ms = interval_ms_map.get(bybit_interval, 4 * 60 * 60 * 1000)
        
        # Calculate how many candles we need
        total_candles_needed = int((end_ms - start_ms) / interval_ms) + 1
        
        # Bybit returns data in reverse chronological order (newest first)
        # We'll paginate backwards from end to start
        current_end = end_ms
        max_iterations = min(100, (total_candles_needed // 200) + 2)  # Prevent infinite loops
        iteration = 0
        
        while current_end > start_ms and iteration < max_iterations:
            iteration += 1
            
            # Request up to 200 candles (Bybit limit)
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": start_ms,
                "end": current_end,
                "limit": 200,
            }
            
            data = self._get("/v5/market/kline", params)
            if not data or "result" not in data or not data["result"].get("list"):
                print(f"[BACKTEST] No data returned from Bybit API for {symbol}")
                break
            
            batch_candles = []
            for item in data["result"]["list"]:
                candle_ts_ms = int(item[0])  # Already in milliseconds from Bybit
                candle_ts = candle_ts_ms // 1000  # Convert to seconds
                
                # Filter by our date range
                if start_ts <= candle_ts < end_ts:
                    batch_candles.append({
                        "timestamp": candle_ts,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    })
            
            if not batch_candles:
                # No more data in range
                break
            
            all_candles.extend(batch_candles)
            
            # Move to next batch - go backwards in time
            # Find the oldest candle timestamp in this batch
            oldest_ts = min(c["timestamp"] for c in batch_candles)
            current_end = oldest_ts * 1000 - 1  # Move before the oldest candle
            
            # If we got less than 200 candles, we might have reached the start
            if len(batch_candles) < 200:
                # Check if we've covered the full range
                if oldest_ts * 1000 <= start_ms:
                    break
        
        # Remove duplicates and sort (oldest first)
        seen = set()
        unique_candles = []
        for c in all_candles:
            if c["timestamp"] not in seen:
                seen.add(c["timestamp"])
                unique_candles.append(c)
        
        unique_candles.sort(key=lambda x: x["timestamp"])
        
        print(f"[BACKTEST] Retrieved {len(unique_candles)} unique candles for {symbol} (range: {start_ts} to {end_ts})")
        
        return unique_candles

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
# Backtesting Module
#############################

class Backtester:
    def __init__(self, client: BybitClient, hist: HistoryProvider):
        self.client = client
        self.hist = hist

    def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Lấy dữ liệu lịch sử từ Bybit trong khoảng thời gian cụ thể"""
        try:
            # Convert dates to timestamps (seconds)
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            # End date should be end of day
            end_ts = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
            
            print(f"[BACKTEST] Fetching {symbol} data from {start_date} to {end_date} (ts: {start_ts} to {end_ts})")
            
            # Use the new method that supports date ranges
            candles = self.client.get_ohlcv_range(symbol, interval, start_ts, end_ts)
            
            print(f"[BACKTEST] Retrieved {len(candles)} candles for {symbol}")
            
            if not candles:
                print(f"[BACKTEST] Warning: No candles retrieved for {symbol}")
                return []
            
            return candles
        except Exception as e:
            print(f"[BACKTEST] Error fetching historical data for {symbol}: {e}")
            return []

    def run_backtest(self, start_date: str, end_date: str, initial_eth: float = 138.0, initial_price: float = 3150.0) -> Dict[str, Any]:
        """Chạy backtest trên khoảng thời gian đã chọn"""
        print(f"[BACKTEST] Starting backtest from {start_date} to {end_date}")
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            now = datetime.now()
            
            if start_dt > now:
                return {"error": f"Start date ({start_date}) is in the future. Please select a past date."}
            
            if end_dt > now:
                return {"error": f"End date ({end_date}) is in the future. Please select a past date."}
            
            if start_dt >= end_dt:
                return {"error": "Start date must be before end date."}
            
            # Check if date range is too large (more than 1 year)
            days_diff = (end_dt - start_dt).days
            if days_diff > 365:
                return {"error": f"Date range is too large ({days_diff} days). Maximum is 365 days."}
            
        except ValueError as e:
            return {"error": f"Invalid date format: {str(e)}. Please use YYYY-MM-DD format."}
        
        # Lấy dữ liệu lịch sử
        print(f"[BACKTEST] Fetching historical data...")
        eth_candles = self.get_historical_data("ETHUSDT", "4h", start_date, end_date)
        btc_candles = self.get_historical_data("BTCUSDT", "4h", start_date, end_date)
        ethbtc_candles = self.get_historical_data("ETHBTC", "4h", start_date, end_date)
        
        print(f"[BACKTEST] Data retrieved - ETH: {len(eth_candles)}, BTC: {len(btc_candles)}, ETHBTC: {len(ethbtc_candles)}")
        
        if not eth_candles:
            return {"error": f"No ETH data found for the selected date range ({start_date} to {end_date}). Please try a different date range."}
        
        if len(eth_candles) < 100:
            return {
                "error": f"Not enough historical data. Found {len(eth_candles)} candles, need at least 100. "
                        f"Please select a longer date range (at least {int(100/6)} days for 4h interval)."
            }
        
        # Initialize position
        position = PositionState(
            spot_eth=initial_eth,
            loan_usdt=0.0,
            avg_entry_eth=initial_price,
        )
        
        # Fetch ETHBTC 1D data once before the loop (for RSI and trend calculation)
        print(f"[BACKTEST] Fetching ETHBTC 1D data for trend analysis...")
        ethbtc_1d_candles = self.get_historical_data("ETHBTC", "1d", start_date, end_date)
        print(f"[BACKTEST] Retrieved {len(ethbtc_1d_candles)} ETHBTC 1D candles")
        
        # Initialize strategy với config mặc định
        conf_builder = DynamicConfigBuilder(self.hist)
        conf = conf_builder.build()
        strategy = EthStrategy(conf)
        
        # Track results
        signals = []
        equity_curve = []
        initial_value = initial_eth * eth_candles[0]["close"]
        equity_curve.append(initial_value)
        
        max_equity = initial_value
        max_drawdown = 0.0
        
        # Process each candle
        print(f"[BACKTEST] Processing {len(eth_candles) - 100} candles...")
        for i in range(100, len(eth_candles)):  # Start from 100 to have enough data for indicators
            candle = eth_candles[i]
            candle_ts = candle["timestamp"]
            
            # Get corresponding BTC and ETHBTC prices
            btc_price = btc_candles[i]["close"] if i < len(btc_candles) else btc_candles[-1]["close"]
            ethbtc_price = ethbtc_candles[i]["close"] if i < len(ethbtc_candles) else ethbtc_candles[-1]["close"]
            
            # Calculate indicators up to this point
            eth_closes = [c["close"] for c in eth_candles[:i+1]]
            btc_closes = [c["close"] for c in btc_candles[:i+1]] if i < len(btc_candles) else [c["close"] for c in btc_candles]
            ethbtc_closes = [c["close"] for c in ethbtc_candles[:i+1]] if i < len(ethbtc_candles) else [c["close"] for c in ethbtc_candles]
            
            eth_rsi_4h = calc_rsi(eth_closes[-80:]) if len(eth_closes) >= 80 else float("nan")
            btc_rsi_4h = calc_rsi(btc_closes[-80:]) if len(btc_closes) >= 80 else float("nan")
            _, _, eth_macd_hist_4h = calc_macd(eth_closes[-80:]) if len(eth_closes) >= 80 else (float("nan"),)*3
            _, _, btc_macd_hist_4h = calc_macd(btc_closes[-80:]) if len(btc_closes) >= 80 else (float("nan"),)*3
            ethbtc_rsi_4h = calc_rsi(ethbtc_closes[-80:]) if len(ethbtc_closes) >= 80 else float("nan")
            
            # Use pre-fetched 1D data for ETHBTC RSI (filter by timestamp)
            ethbtc_1d_closes = [c["close"] for c in ethbtc_1d_candles if c["timestamp"] <= candle_ts]
            ethbtc_rsi_1d = calc_rsi(ethbtc_1d_closes) if len(ethbtc_1d_closes) >= 14 else float("nan")
            
            # Determine ETHBTC trend
            if len(ethbtc_1d_closes) >= 200:
                ma200 = sum(ethbtc_1d_closes[-200:]) / 200
                last_price = ethbtc_1d_closes[-1]
                if last_price > ma200 and ethbtc_rsi_1d > 50:
                    ethbtc_trend = "bull"
                elif last_price < ma200 and ethbtc_rsi_1d < 50:
                    ethbtc_trend = "bear"
                else:
                    ethbtc_trend = "neutral"
            else:
                ethbtc_trend = "neutral"
            
            # Update config periodically (every ~24 candles = ~4 days)
            if i % 24 == 0:  # Update config every ~4 days
                try:
                    # Create a mock history provider for this point in time
                    # For simplicity, use current data up to this point
                    conf = conf_builder.build()
                    strategy.update_conf(conf)
                except Exception as e:
                    print(f"[BACKTEST] Config update error at candle {i}: {e}")
                    pass
            
            # Build market state
            market_state = MarketState(
                eth_price=candle["close"],
                btc_price=btc_price,
                ethbtc_price=ethbtc_price,
                eth_rsi_4h=eth_rsi_4h,
                btc_rsi_4h=btc_rsi_4h,
                eth_macd_hist_4h=eth_macd_hist_4h,
                btc_macd_hist_4h=btc_macd_hist_4h,
                ethbtc_rsi_4h=ethbtc_rsi_4h,
                ethbtc_rsi_1d=ethbtc_rsi_1d,
                ethbtc_trend=ethbtc_trend,
            )
            
            # Check for signals (same priority as live bot)
            signal = None
            signal = strategy.should_repay_loan(market_state, position)
            if not signal:
                signal = strategy.should_sell_spot(market_state, position)
            if not signal:
                signal = strategy.should_buyback_spot(market_state, position)
            if not signal:
                signal = strategy.should_open_loan(market_state, position)
            
            # Debug: Log why no signal (only for first few candles and periodically)
            if not signal and (i < 110 or i % 50 == 0):
                debug_info = {
                    "candle": i,
                    "eth_price": market_state.eth_price,
                    "eth_rsi_4h": market_state.eth_rsi_4h,
                    "eth_macd_hist_4h": market_state.eth_macd_hist_4h,
                    "btc_price": market_state.btc_price,
                    "ethbtc_price": market_state.ethbtc_price,
                    "ethbtc_trend": market_state.ethbtc_trend,
                    "position_eth": position.spot_eth,
                    "loan_usdt": position.loan_usdt,
                    "last_sell_size": position.last_sell_size,
                }
                print(f"[BACKTEST DEBUG] No signal at candle {i}: {debug_info}")
            
            if signal:
                # Execute signal
                action = signal["action"]
                
                if action == "SELL_SPOT_ETH":
                    size = signal["size"]
                    position.spot_eth = max(0.0, position.spot_eth - size)
                    position.last_sell_price = market_state.eth_price
                    position.last_sell_size = size
                    strategy.last_sell_ts = candle_ts
                    
                elif action == "BUYBACK_SPOT_ETH":
                    size = signal["size"]
                    position.spot_eth += size
                    position.last_sell_price = None
                    position.last_sell_size = None
                    
                elif action == "OPEN_LOAN_BUY_ETH":
                    loan = signal["loan_amount"]
                    position.loan_usdt += loan
                    bought_eth = loan / market_state.eth_price
                    position.spot_eth += bought_eth
                    
                elif action == "REPAY_LOAN_SELL_ETH":
                    size = signal["size"]
                    position.spot_eth = max(0.0, position.spot_eth - size)
                    position.loan_usdt = 0.0
                
                signals.append({
                    "timestamp": candle_ts,
                    "action": action,
                    "eth_price": market_state.eth_price,
                    "signal": signal,
                })
            
            # Calculate current equity
            current_value = position.spot_eth * market_state.eth_price - position.loan_usdt
            equity_curve.append(current_value)
            
            if current_value > max_equity:
                max_equity = current_value
            
            drawdown = ((max_equity - current_value) / max_equity) * 100 if max_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate final metrics
        final_value = equity_curve[-1] if equity_curve else initial_value
        return_pct = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        # Calculate win rate (simplified - based on profitable trades)
        profitable_trades = 0
        total_trades = len(signals)
        if total_trades > 0:
            # Simple heuristic: if final value > initial, consider it winning
            if final_value > initial_value:
                profitable_trades = total_trades * 0.6  # Rough estimate
            else:
                profitable_trades = total_trades * 0.4
            win_rate = (profitable_trades / total_trades) * 100
        else:
            win_rate = 0.0
        
        # Sample equity curve if too large (for performance and response size)
        sampled_equity_curve = equity_curve
        if len(equity_curve) > 1000:
            # Sample every Nth point to keep max 1000 points
            step = len(equity_curve) // 1000
            sampled_equity_curve = [equity_curve[i] for i in range(0, len(equity_curve), step)]
            # Always include the last point
            if sampled_equity_curve[-1] != equity_curve[-1]:
                sampled_equity_curve.append(equity_curve[-1])
            print(f"[BACKTEST] Sampled equity curve from {len(equity_curve)} to {len(sampled_equity_curve)} points")
        
        print(f"[BACKTEST] Backtest completed: {total_trades} signals, return: {return_pct:.2f}%")
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_value": initial_value,
            "final_value": final_value,
            "return_pct": return_pct,
            "max_drawdown": max_drawdown,
            "total_signals": total_trades,
            "win_rate": win_rate,
            "equity_curve": sampled_equity_curve,
            "signals": signals[-20:],  # Last 20 signals
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
            telegram_msg = (
                f"ETH Bot – Config Updated\n"
            )
            telegram_notify(telegram_msg)

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
    bot_pos = bot.position if hasattr(bot, 'position') else None

    def fmt_float(val, digits=2):
        return f"{val:.{digits}f}" if isinstance(val, (int, float)) and not isnan(val) else "N/A"

    def get_rsi_color(rsi):
        if isnan(rsi):
            return "#888"
        if rsi >= 70:
            return "#ef4444"
        elif rsi <= 30:
            return "#22c55e"
        return "#f0f0f0"

    def get_trend_color(trend):
        colors = {"bull": "#22c55e", "bear": "#ef4444", "neutral": "#888"}
        return colors.get(trend, "#888")

    # Prepare signals data for chart
    signals_data = []
    for s in RECENT_SIGNALS[-50:]:
        signals_data.append({
            "time": s["ts"],
            "action": s["action"],
            "price": s["market"].get("eth_price", 0)
        })

    # Calculate position metrics
    position_value = 0
    pnl_pct = 0
    if bot_pos and m.get('eth_price'):
        position_value = bot_pos.spot_eth * m.get('eth_price', 0)
        if bot_pos.avg_entry_eth > 0:
            pnl_pct = ((m.get('eth_price', 0) - bot_pos.avg_entry_eth) / bot_pos.avg_entry_eth) * 100

    html_signals = ""
    action_colors = {
        "SELL_SPOT_ETH": "#ef4444",
        "BUYBACK_SPOT_ETH": "#22c55e",
        "OPEN_LOAN_BUY_ETH": "#3b82f6",
        "REPAY_LOAN_SELL_ETH": "#f59e0b"
    }
    for s in reversed(RECENT_SIGNALS[-30:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["ts"]))
        mk = s["market"]
        action = s['action']
        color = action_colors.get(action, "#888")
        html_signals += f"""
        <tr>
          <td>{ts}</td>
          <td><span style="color: {color}; font-weight: bold;">{action}</span></td>
          <td>{fmt_float(mk.get('eth_price'), 2)}</td>
          <td>{fmt_float(mk.get('btc_price'), 2)}</td>
          <td>{fmt_float(mk.get('ethbtc_price'), 6)}</td>
          <td>{fmt_float(mk.get('eth_rsi_4h'), 2)}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>ETH Strategy Bot Dashboard</title>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
      <style>
        * {{
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }}
        body {{
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background: linear-gradient(135deg, #0b1020 0%, #1a1f3a 100%);
          color: #f0f0f0;
          padding: 20px;
          min-height: 100vh;
        }}
        .container {{
          max-width: 1400px;
          margin: 0 auto;
        }}
        .header {{
          text-align: center;
          margin-bottom: 30px;
          padding: 20px;
          background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
          border-radius: 15px;
          box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .header h1 {{
          color: #ffd166;
          font-size: 2.5em;
          margin-bottom: 10px;
          text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        .header .subtitle {{
          color: #a0a0a0;
          font-size: 1.1em;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
          margin-bottom: 20px;
        }}
        .card {{
          background: linear-gradient(135deg, #141a33 0%, #1e2540 100%);
          border-radius: 15px;
          padding: 20px;
          box-shadow: 0 8px 32px rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.1);
          transition: transform 0.3s, box-shadow 0.3s;
        }}
        .card:hover {{
          transform: translateY(-5px);
          box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }}
        .card h2 {{
          color: #ffd166;
          margin-bottom: 15px;
          font-size: 1.4em;
          border-bottom: 2px solid rgba(255,209,102,0.3);
          padding-bottom: 10px;
        }}
        .metric {{
          display: flex;
          justify-content: space-between;
          padding: 10px 0;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric:last-child {{
          border-bottom: none;
        }}
        .metric-label {{
          color: #a0a0a0;
        }}
        .metric-value {{
          font-weight: bold;
          font-size: 1.1em;
        }}
        .metric-value.positive {{
          color: #22c55e;
        }}
        .metric-value.negative {{
          color: #ef4444;
        }}
        .rsi-indicator {{
          display: inline-block;
          width: 12px;
          height: 12px;
          border-radius: 50%;
          margin-right: 8px;
        }}
        .trend-badge {{
          display: inline-block;
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 0.85em;
          font-weight: bold;
        }}
        .trend-bull {{
          background: rgba(34,197,94,0.2);
          color: #22c55e;
        }}
        .trend-bear {{
          background: rgba(239,68,68,0.2);
          color: #ef4444;
        }}
        .trend-neutral {{
          background: rgba(136,136,136,0.2);
          color: #888;
        }}
        .chart-container {{
          position: relative;
          height: 300px;
          margin-top: 15px;
        }}
        .signals-table {{
          width: 100%;
          border-collapse: collapse;
          margin-top: 15px;
        }}
        .signals-table th {{
          background: rgba(255,209,102,0.2);
          color: #ffd166;
          padding: 12px;
          text-align: left;
          font-weight: 600;
          border-bottom: 2px solid rgba(255,209,102,0.3);
        }}
        .signals-table td {{
          padding: 10px 12px;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .signals-table tr:hover {{
          background: rgba(255,255,255,0.05);
        }}
        .config-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
          margin-top: 15px;
        }}
        .config-item {{
          background: rgba(0,0,0,0.2);
          padding: 10px;
          border-radius: 8px;
          border-left: 3px solid #ffd166;
        }}
        .config-label {{
          font-size: 0.85em;
          color: #a0a0a0;
          margin-bottom: 5px;
        }}
        .config-value {{
          font-weight: bold;
          color: #f0f0f0;
        }}
        .nav-tabs {{
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
          border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        .nav-tab {{
          padding: 12px 24px;
          background: transparent;
          border: none;
          color: #a0a0a0;
          cursor: pointer;
          font-size: 1em;
          border-bottom: 2px solid transparent;
          transition: all 0.3s;
        }}
        .nav-tab.active {{
          color: #ffd166;
          border-bottom-color: #ffd166;
        }}
        .tab-content {{
          display: none;
        }}
        .tab-content.active {{
          display: block;
        }}
        .status-indicator {{
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          margin-right: 8px;
          animation: pulse 2s infinite;
        }}
        .status-online {{
          background: #22c55e;
        }}
        @keyframes pulse {{
          0%, 100% {{ opacity: 1; }}
          50% {{ opacity: 0.5; }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>🚀 ETH Strategy Bot</h1>
          <div class="subtitle">
            <span class="status-indicator status-online"></span>
            Live Trading Dashboard • Bybit Integration
          </div>
        </div>

        <div class="nav-tabs">
          <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
          <button class="nav-tab" onclick="showTab('signals')">Signals</button>
          <button class="nav-tab" onclick="showTab('config')">Config</button>
          <button class="nav-tab" onclick="showTab('backtest')">Backtest</button>
        </div>

        <div id="overview" class="tab-content active">
          <div class="grid">
            <div class="card">
              <h2>💰 Position</h2>
              <div class="metric">
                <span class="metric-label">Spot ETH</span>
                <span class="metric-value">{fmt_float(bot_pos.spot_eth if bot_pos else 0, 4)} ETH</span>
              </div>
              <div class="metric">
                <span class="metric-label">Position Value</span>
                <span class="metric-value">${fmt_float(position_value, 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Loan USDT</span>
                <span class="metric-value">{fmt_float(bot_pos.loan_usdt if bot_pos else 0, 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Avg Entry</span>
                <span class="metric-value">${fmt_float(bot_pos.avg_entry_eth if bot_pos else 0, 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">P&L %</span>
                <span class="metric-value {'positive' if pnl_pct >= 0 else 'negative'}">
                  {fmt_float(pnl_pct, 2)}%
                </span>
              </div>
            </div>

            <div class="card">
              <h2>📊 Market Prices</h2>
              <div class="metric">
                <span class="metric-label">ETH/USDT</span>
                <span class="metric-value">${fmt_float(m.get('eth_price'), 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">BTC/USDT</span>
                <span class="metric-value">${fmt_float(m.get('btc_price'), 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">ETH/BTC</span>
                <span class="metric-value">{fmt_float(m.get('ethbtc_price'), 6)}</span>
              </div>
            </div>

            <div class="card">
              <h2>📈 Indicators</h2>
              <div class="metric">
                <span class="metric-label">ETH RSI 4H</span>
                <span class="metric-value">
                  <span class="rsi-indicator" style="background: {get_rsi_color(m.get('eth_rsi_4h'))};"></span>
                  {fmt_float(m.get('eth_rsi_4h'), 2)}
                </span>
              </div>
              <div class="metric">
                <span class="metric-label">BTC RSI 4H</span>
                <span class="metric-value">
                  <span class="rsi-indicator" style="background: {get_rsi_color(m.get('btc_rsi_4h'))};"></span>
                  {fmt_float(m.get('btc_rsi_4h'), 2)}
                </span>
              </div>
              <div class="metric">
                <span class="metric-label">ETHBTC RSI 4H</span>
                <span class="metric-value">{fmt_float(m.get('ethbtc_rsi_4h'), 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">ETHBTC RSI 1D</span>
                <span class="metric-value">{fmt_float(m.get('ethbtc_rsi_1d'), 2)}</span>
              </div>
              <div class="metric">
                <span class="metric-label">ETHBTC Trend</span>
                <span class="metric-value">
                  <span class="trend-badge trend-{m.get('ethbtc_trend', 'neutral')}">
                    {m.get('ethbtc_trend', 'N/A').upper()}
                  </span>
                </span>
              </div>
            </div>
          </div>

          <div class="card" style="margin-top: 20px;">
            <h2>📉 Price Chart (Last 50 Signals)</h2>
            <div class="chart-container">
              <canvas id="priceChart"></canvas>
            </div>
          </div>
        </div>

        <div id="signals" class="tab-content">
          <div class="card">
            <h2>📡 Recent Trading Signals</h2>
            <table class="signals-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Action</th>
                  <th>ETH Price</th>
                  <th>BTC Price</th>
                  <th>ETH/BTC</th>
                  <th>RSI 4H</th>
                </tr>
              </thead>
              <tbody>
                {html_signals if html_signals else "<tr><td colspan='6' style='text-align: center; padding: 20px; color: #888;'>No signals yet</td></tr>"}
              </tbody>
            </table>
          </div>
        </div>

        <div id="config" class="tab-content">
          <div class="card">
            <h2>⚙️ Dynamic Configuration</h2>
            <div class="config-grid">
              <div class="config-item">
                <div class="config-label">ETH RSI Overbought (4H)</div>
                <div class="config-value">{fmt_float(c.get('eth_rsi_overbought_4h'), 2)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">ETH RSI Oversold (4H)</div>
                <div class="config-value">{fmt_float(c.get('eth_rsi_oversold_4h'), 2)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">BTC Dump Level</div>
                <div class="config-value">${fmt_float(c.get('btc_dump_level'), 2)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">Max LTV</div>
                <div class="config-value">{fmt_float(c.get('max_ltv', 0) * 100, 1)}%</div>
              </div>
              <div class="config-item">
                <div class="config-label">Buyback Zone Low</div>
                <div class="config-value">${fmt_float(c.get('buyback_zone', (0,0))[0] if isinstance(c.get('buyback_zone'), tuple) else 0, 2)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">Buyback Zone High</div>
                <div class="config-value">${fmt_float(c.get('buyback_zone', (0,0))[1] if isinstance(c.get('buyback_zone'), tuple) else 0, 2)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">ETHBTC Breakdown</div>
                <div class="config-value">{fmt_float(c.get('ethbtc_breakdown'), 6)}</div>
              </div>
              <div class="config-item">
                <div class="config-label">ETH MA200 (1D)</div>
                <div class="config-value">${fmt_float(c.get('eth_ma200_1d'), 2)}</div>
              </div>
            </div>
          </div>
        </div>

        <div id="backtest" class="tab-content">
          <div class="card">
            <h2>🧪 Backtesting</h2>
            <p style="margin-bottom: 15px; color: #a0a0a0;">
              Test your strategy on historical data. Select a date range and run backtest.
            </p>
            <div style="display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
              <div>
                <label style="display: block; margin-bottom: 5px; color: #a0a0a0;">Start Date:</label>
                <input type="date" id="backtestStart" style="padding: 8px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #f0f0f0;">
              </div>
              <div>
                <label style="display: block; margin-bottom: 5px; color: #a0a0a0;">End Date:</label>
                <input type="date" id="backtestEnd" style="padding: 8px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #f0f0f0;">
              </div>
              <div style="display: flex; align-items: flex-end;">
                <button onclick="runBacktest()" style="padding: 10px 20px; background: #ffd166; color: #0b1020; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; transition: all 0.3s;">
                  Run Backtest
                </button>
              </div>
            </div>
            <div id="backtestResults" style="margin-top: 20px;"></div>
            <div class="chart-container" id="backtestChartContainer" style="display: none;">
              <canvas id="backtestChart"></canvas>
            </div>
          </div>
        </div>
      </div>

      <script>
        // Tab switching
        function showTab(tabName) {{
          document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
          document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
          document.getElementById(tabName).classList.add('active');
          event.target.classList.add('active');
        }}

        // Price chart
        const signalsData = {json.dumps(signals_data)};
        let priceChart = null;
        if (signalsData && signalsData.length > 0) {{
          const ctx = document.getElementById('priceChart').getContext('2d');
          priceChart = new Chart(ctx, {{
            type: 'line',
            data: {{
              labels: signalsData.map(s => new Date(s.time * 1000).toLocaleString()),
              datasets: [{{
                label: 'ETH Price at Signal',
                data: signalsData.map(s => s.price),
                borderColor: '#ffd166',
                backgroundColor: 'rgba(255, 209, 102, 0.1)',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: signalsData.map(s => {{
                  const colors = {{
                    'SELL_SPOT_ETH': '#ef4444',
                    'BUYBACK_SPOT_ETH': '#22c55e',
                    'OPEN_LOAN_BUY_ETH': '#3b82f6',
                    'REPAY_LOAN_SELL_ETH': '#f59e0b'
                  }};
                  return colors[s.action] || '#888';
                }}),
                tension: 0.4
              }}]
            }},
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {{
                legend: {{
                  labels: {{ color: '#f0f0f0' }}
                }}
              }},
              scales: {{
                x: {{
                  ticks: {{ color: '#a0a0a0' }},
                  grid: {{ color: 'rgba(255,255,255,0.1)' }}
                }},
                y: {{
                  ticks: {{ color: '#a0a0a0' }},
                  grid: {{ color: 'rgba(255,255,255,0.1)' }}
                }}
              }}
            }}
          }});
        }} else {{
          document.getElementById('priceChart').parentElement.innerHTML = '<p style="text-align: center; color: #888; padding: 40px;">No signals data available yet</p>';
        }}

        // Backtest function
        async function runBacktest() {{
          const start = document.getElementById('backtestStart').value;
          const end = document.getElementById('backtestEnd').value;
          
          if (!start || !end) {{
            alert('Please select both start and end dates');
            return;
          }}

          const resultsDiv = document.getElementById('backtestResults');
          resultsDiv.innerHTML = '<p style="color: #ffd166;">⏳ Running backtest... This may take a minute. Please wait.</p>';

          try {{
            console.log(`[BACKTEST] Starting backtest from ${{start}} to ${{end}}`);
            
            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout
            
            const response = await fetch(`/api/backtest?start=${{start}}&end=${{end}}`, {{
              signal: controller.signal
            }});
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {{
              const errorText = await response.text();
              throw new Error(`HTTP error! status: ${{response.status}}, message: ${{errorText}}`);
            }}
            
            const data = await response.json();
            console.log('[BACKTEST] Response received, total_signals:', data.total_signals, 'equity_curve_length:', data.equity_curve?.length);
            
            if (data.error) {{
              resultsDiv.innerHTML = `
                <div style="background: rgba(239,68,68,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444;">
                  <p style="color: #ef4444; font-weight: bold; margin-bottom: 10px;">❌ Error</p>
                  <p style="color: #f0f0f0;">${{data.error}}</p>
                  <p style="color: #a0a0a0; font-size: 0.9em; margin-top: 10px;">
                    💡 Tip: Make sure the dates are in the past and the range is at least 17 days (for 4h interval).
                  </p>
                </div>
              `;
              document.getElementById('backtestChartContainer').style.display = 'none';
              return;
            }}

            // Display results
            resultsDiv.innerHTML = `
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                <div class="config-item">
                  <div class="config-label">Total Signals</div>
                  <div class="config-value">${{data.total_signals}}</div>
                </div>
                <div class="config-item">
                  <div class="config-label">Final Value</div>
                  <div class="config-value">$${{data.final_value.toFixed(2)}}</div>
                </div>
                <div class="config-item">
                  <div class="config-label">Return %</div>
                  <div class="config-value" style="color: ${{data.return_pct >= 0 ? '#22c55e' : '#ef4444'}}">
                    ${{data.return_pct.toFixed(2)}}%
                  </div>
                </div>
                <div class="config-item">
                  <div class="config-label">Max Drawdown</div>
                  <div class="config-value" style="color: #ef4444;">${{data.max_drawdown.toFixed(2)}}%</div>
                </div>
                <div class="config-item">
                  <div class="config-label">Win Rate</div>
                  <div class="config-value">${{data.win_rate.toFixed(1)}}%</div>
                </div>
              </div>
            `;

            // Show chart
            if (data.equity_curve && data.equity_curve.length > 0) {{
              document.getElementById('backtestChartContainer').style.display = 'block';
              const backtestCtx = document.getElementById('backtestChart').getContext('2d');
              
              if (window.backtestChartInstance) {{
                window.backtestChartInstance.destroy();
              }}

              // Sample equity curve if too large (for performance)
              let chartData = data.equity_curve;
              let chartLabels = chartData.map((_, i) => `Candle ${{i+1}}`);
              
              if (chartData.length > 500) {{
                // Sample every Nth point
                const step = Math.ceil(chartData.length / 500);
                chartData = chartData.filter((_, i) => i % step === 0);
                chartLabels = chartData.map((_, i) => `Candle ${{i * step + 1}}`);
                console.log(`[BACKTEST] Sampled equity curve from ${{data.equity_curve.length}} to ${{chartData.length}} points`);
              }}

              window.backtestChartInstance = new Chart(backtestCtx, {{
                type: 'line',
                data: {{
                  labels: chartLabels,
                  datasets: [{{
                    label: 'Equity Curve',
                    data: chartData,
                    borderColor: '#ffd166',
                    backgroundColor: 'rgba(255, 209, 102, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                  }}]
                }},
              options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                  legend: {{
                    labels: {{ color: '#f0f0f0' }}
                  }}
                }},
                scales: {{
                  x: {{
                    ticks: {{ color: '#a0a0a0' }},
                    grid: {{ color: 'rgba(255,255,255,0.1)' }}
                  }},
                  y: {{
                    ticks: {{ color: '#a0a0a0' }},
                    grid: {{ color: 'rgba(255,255,255,0.1)' }}
                  }}
                }}
              }}
            }});
            }} else {{
              console.warn('[BACKTEST] No equity curve data to display');
              document.getElementById('backtestChartContainer').style.display = 'none';
            }}
          }} catch (error) {{
            console.error('[BACKTEST] Error:', error);
            resultsDiv.innerHTML = `
              <div style="background: rgba(239,68,68,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444;">
                <p style="color: #ef4444; font-weight: bold; margin-bottom: 10px;">❌ Error</p>
                <p style="color: #f0f0f0;">${{error.message}}</p>
                <p style="color: #a0a0a0; font-size: 0.9em; margin-top: 10px;">
                  Please check the browser console for more details.
                </p>
              </div>
            `;
            document.getElementById('backtestChartContainer').style.display = 'none';
          }}
        }}

        // Auto-refresh every 60 seconds
        setInterval(() => {{
          location.reload();
        }}, 60000);
      </script>
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


@app.get("/api/backtest", response_class=JSONResponse)
def api_backtest(start: str, end: str):
    """API endpoint để chạy backtest"""
    try:
        backtester = Backtester(bot.client, bot.hist)
        results = backtester.run_backtest(
            start_date=start,
            end_date=end,
            initial_eth=INITIAL_SPOT_ETH,
            initial_price=AVG_ENTRY_ETH
        )
        
        if "error" in results:
            return JSONResponse(content={"error": results["error"]}, status_code=400)
        
        safe = sanitize_for_json(results)
        return safe
    except Exception as e:
        print(f"[BACKTEST ERROR]", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
