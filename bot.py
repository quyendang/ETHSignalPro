import os
import time
import threading
import json
import hmac
import hashlib
from dataclasses import dataclass, asdict
from math import isnan
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import math
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

#############################
# ENV & GLOBAL CONFIG
#############################

INITIAL_ETH = float(os.getenv("INITIAL_ETH", "100"))
INITIAL_ENTRY_PRICE = float(os.getenv("INITIAL_ENTRY_PRICE", "3150"))
MAX_LTV = float(os.getenv("MAX_LTV", "0.15"))  # 15% max LTV

PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "eth_rotation_signals")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# In-memory state
RECENT_SIGNALS: List[Dict[str, Any]] = []
MAX_SIGNALS = 200

LAST_MARKET_STATE: Dict[str, Any] = {}
LAST_CONFIG: Dict[str, Any] = {}

#############################
# Helpers: Notification
#############################

def pushover_notify(title: str, message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
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
        print(f"[PUSHOVER ERROR] {e}")

def telegram_notify(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def supabase_log_signal(action: str, market_state: Dict, signal: Dict):
    if not SUPABASE_URL or not SUPABASE_KEY:
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
        "eth_price": market_state.get("eth_price", 0),
        "btc_price": market_state.get("btc_price", 0),
        "signal_data": signal,
        "timestamp": int(time.time()),
    }
    try:
        requests.post(url, json=data, headers=headers, timeout=10)
    except Exception as e:
        print(f"[SUPABASE ERROR] {e}")

#############################
# Technical Indicators
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

#############################
# Binance Client (for better historical data)
#############################

class BinanceClient:
    BASE_URL = "https://api.binance.com"

    def __init__(self):
        self.use_auth = False  # Binance public market data doesn't need auth

    def _map_interval(self, interval: str) -> str:
        """Map interval to Binance format"""
        mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M",
        }
        return mapping.get(interval, interval)

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL + path
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[BINANCE ERROR] {e}")
            return None

    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get latest price"""
        data = self._get("/api/v3/ticker/price", {"symbol": symbol})
        if not data or "price" not in data:
            return {"last": float("nan")}
        return {"last": float(data["price"])}

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent OHLCV data"""
        bybit_interval = self._map_interval(interval)
        limit = min(limit, 1000)  # Binance max is 1000
        
        data = self._get("/api/v3/klines", {
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit,
        })
        
        if not data or not isinstance(data, list):
            return []
        
        candles = []
        for item in data:
            candles.append({
                "timestamp": int(item[0]) // 1000,  # Convert ms to seconds
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            })
        return candles

    def get_ohlcv_range(self, symbol: str, interval: str, start_ts: int, end_ts: int) -> List[Dict[str, Any]]:
        """Get OHLCV data in time range (timestamps in seconds)"""
        bybit_interval = self._map_interval(interval)
        all_candles = []
        
        # Binance uses milliseconds
        start_ms = start_ts * 1000
        end_ms = end_ts * 1000
        current_start = start_ms
        
        # Binance limit is 1000 candles per request
        limit = 1000
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        print(f"[BINANCE] Fetching {symbol} {interval} from {datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d')}")
        
        while current_start < end_ms and iteration < max_iterations:
            iteration += 1
            
            params = {
                "symbol": symbol,
                "interval": bybit_interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }
            
            data = self._get("/api/v3/klines", params)
            
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            
            batch = []
            for item in data:
                ts_ms = int(item[0])
                if start_ms <= ts_ms < end_ms:
                    batch.append({
                        "timestamp": ts_ms // 1000,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    })
            
            if not batch:
                break
            
            all_candles.extend(batch)
            
            # Move to next batch (last candle time + 1ms)
            last_ts = batch[-1]["timestamp"] * 1000
            current_start = last_ts + 1
            
            print(f"[BINANCE] Fetched {len(batch)} candles (total: {len(all_candles)}, latest: {datetime.fromtimestamp(batch[-1]['timestamp']).strftime('%Y-%m-%d %H:%M')})")
            
            if len(data) < limit:
                # Last batch
                break
        
        all_candles.sort(key=lambda x: x["timestamp"])
        print(f"[BINANCE] Retrieved {len(all_candles)} total candles for {symbol} {interval}")
        return all_candles

#############################
# Bybit Client (kept for compatibility)
#############################

class BybitClient:
    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key or BYBIT_API_KEY
        self.api_secret = api_secret or BYBIT_API_SECRET
        self.use_auth = bool(self.api_key and self.api_secret)

    def _generate_signature(self, params: Dict[str, Any], timestamp: str, recv_window: str = "5000") -> str:
        """Generate HMAC SHA256 signature for Bybit API v5"""
        # Sort parameters alphabetically
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Bybit API v5 signature format: timestamp + api_key + recv_window + query_string
        payload = f"{timestamp}{self.api_key}{recv_window}{query_string}"
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def _get(self, path: str, params: Dict[str, Any], use_auth: bool = False) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL + path
        headers = {}
        
        # Add authentication if requested and credentials are available
        # Note: Market data endpoints (/v5/market/*) are typically public and don't require auth
        # However, authenticated requests may have higher rate limits
        if use_auth and self.use_auth:
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"
            
            # Generate signature from params (don't include api_key, timestamp, recv_window in signature params)
            signature = self._generate_signature(params, timestamp, recv_window)
            
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
            }
        
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                ret_msg = data.get("retMsg", "Unknown error")
                print(f"[BYBIT ERROR] {ret_msg} (retCode: {data.get('retCode')})")
                return None
            return data
        except Exception as e:
            print(f"[BYBIT ERROR] {e}")
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
            "4h": "240",
            "1d": "D",
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

#############################
# Data Structures
#############################

@dataclass
class MarketState:
    eth_price: float
    btc_price: float
    
    eth_rsi_4h: float
    eth_rsi_1d: float
    btc_rsi_4h: float
    btc_rsi_1d: float
    
    eth_macd_hist: float
    btc_macd_hist: float

@dataclass
class PositionState:
    spot_eth: float
    loan_usdt: float
    avg_entry_price: float

#############################
# Dynamic Config Builder
#############################

class DynamicConfigBuilder:
    def __init__(self, client):
        self.client = client

    def build(self) -> Dict[str, Any]:
        # Get historical data
        eth_4h = self.client.get_ohlcv("ETHUSDT", "4h", 200)
        eth_1d = self.client.get_ohlcv("ETHUSDT", "1d", 200)
        btc_4h = self.client.get_ohlcv("BTCUSDT", "4h", 200)
        btc_1d = self.client.get_ohlcv("BTCUSDT", "1d", 200)
        
        if not eth_4h or not eth_1d:
            return self._default_config()
        
        eth_closes_4h = [c["close"] for c in eth_4h]
        eth_closes_1d = [c["close"] for c in eth_1d]
        btc_closes_4h = [c["close"] for c in btc_4h] if btc_4h else []
        btc_closes_1d = [c["close"] for c in btc_1d] if btc_1d else []
        
        # Calculate RSI series for dynamic thresholds
        eth_rsi_4h_series = []
        eth_rsi_1d_series = []
        for i in range(50, len(eth_closes_4h)):
            eth_rsi_4h_series.append(calc_rsi(eth_closes_4h[:i+1]))
        for i in range(50, len(eth_closes_1d)):
            eth_rsi_1d_series.append(calc_rsi(eth_closes_1d[:i+1]))
        
        # Dynamic RSI thresholds (20th and 80th percentile)
        eth_rsi_4h_low = np.percentile([r for r in eth_rsi_4h_series if not isnan(r)], 20) if eth_rsi_4h_series else 30
        eth_rsi_4h_high = np.percentile([r for r in eth_rsi_4h_series if not isnan(r)], 80) if eth_rsi_4h_series else 70
        eth_rsi_1d_low = np.percentile([r for r in eth_rsi_1d_series if not isnan(r)], 20) if eth_rsi_1d_series else 30
        eth_rsi_1d_high = np.percentile([r for r in eth_rsi_1d_series if not isnan(r)], 80) if eth_rsi_1d_series else 70
        
        # Calculate buy/sell zones based on recent price action
        recent_prices = eth_closes_4h[-60:]  # Last 60 candles (10 days)
        price_low = min(recent_prices)
        price_high = max(recent_prices)
        price_range = price_high - price_low
        
        # Buy zone: Lower 30% of range
        buy_zone_low = price_low
        buy_zone_high = price_low + price_range * 0.3
        
        # Sell zone: Upper 30% of range
        sell_zone_low = price_high - price_range * 0.3
        sell_zone_high = price_high
        
        # Current price position in range
        current_price = eth_closes_4h[-1]
        price_position = (current_price - price_low) / price_range if price_range > 0 else 0.5
        
        return {
            "eth_rsi_4h_low": float(eth_rsi_4h_low),
            "eth_rsi_4h_high": float(eth_rsi_4h_high),
            "eth_rsi_1d_low": float(eth_rsi_1d_low),
            "eth_rsi_1d_high": float(eth_rsi_1d_high),
            "buy_zone": (float(buy_zone_low), float(buy_zone_high)),
            "sell_zone": (float(sell_zone_low), float(sell_zone_high)),
            "price_range": (float(price_low), float(price_high)),
            "current_price_position": float(price_position),
        }
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "eth_rsi_4h_low": 30,
            "eth_rsi_4h_high": 70,
            "eth_rsi_1d_low": 30,
            "eth_rsi_1d_high": 70,
            "buy_zone": (0, 0),
            "sell_zone": (0, 0),
            "price_range": (0, 0),
            "current_price_position": 0.5,
        }

#############################
# Rotation Strategy
#############################

class EthRotationStrategy:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.last_eth_macd_hist = None
        self.last_btc_macd_hist = None
        # Risk management: track current cycle state
        self.current_cycle_loans = 0  # Number of loans in current cycle
        self.current_cycle_first_price = None  # First buy price in current cycle
        self.max_loans_per_cycle = 3  # Maximum loans per cycle
        self.max_cycle_loan_amount = 50_000  # Maximum total loan per cycle
        self.stop_loss_pct = -8.0  # Stop loss at -8% from avg buy price

    def update_conf(self, conf: Dict[str, Any]):
        self.conf = conf

    def max_safe_loan(self, m: MarketState, p: PositionState) -> float:
        """Calculate max safe loan amount with 15% LTV"""
        if p.spot_eth <= 0:
            return 0.0
        collateral_value = p.spot_eth * m.eth_price
        current_ltv = p.loan_usdt / collateral_value if collateral_value > 0 else 0.0
        available_ltv = MAX_LTV - current_ltv
        if available_ltv <= 0:
            return 0.0
        return collateral_value * available_ltv

    def should_buy_with_loan(self, m: MarketState, p: PositionState, current_cycle_info: Optional[Dict] = None) -> Optional[Dict]:
        """Buy ETH using margin loan when conditions are favorable"""
        # Risk management: Check cycle limits
        if current_cycle_info:
            cycle_loans = len(current_cycle_info.get("loans", []))
            cycle_total_loan = current_cycle_info.get("total_loan", 0.0)
            cycle_avg_price = current_cycle_info.get("avg_buy_price", 0.0)
            
            # Limit 1: Maximum loans per cycle
            if cycle_loans >= self.max_loans_per_cycle:
                return None
            
            # Limit 2: Maximum total loan per cycle
            if cycle_total_loan >= self.max_cycle_loan_amount:
                return None
            
            # Limit 3: Don't add more loans if price has increased too much from first buy
            if cycle_avg_price > 0:
                price_increase_pct = ((m.eth_price - cycle_avg_price) / cycle_avg_price) * 100
                if price_increase_pct > 5.0:  # Don't buy more if price already up 5%+
                    return None
        
        # Check if we can take more loan
        max_loan = self.max_safe_loan(m, p)
        if max_loan < 1000:  # Minimum loan amount
            return None
        
        buy_low, buy_high = self.conf.get("buy_zone", (0, 0))
        if buy_low <= 0 or buy_high <= 0:
            return None
        
        # Price should be in buy zone
        if not (buy_low <= m.eth_price <= buy_high):
            return None
        
        # RSI conditions: ETH RSI 4h should be low (oversold)
        rsi_low = self.conf.get("eth_rsi_4h_low", 30)
        if isnan(m.eth_rsi_4h) or m.eth_rsi_4h > rsi_low + 10:  # Not oversold enough
            return None
        
        # MACD: ETH MACD hist should be negative or turning positive
        if not isnan(m.eth_macd_hist):
            if m.eth_macd_hist > 50:  # Too bullish, wait
                return None
            # Prefer when MACD is turning from negative to positive
            if self.last_eth_macd_hist is not None:
                if m.eth_macd_hist < self.last_eth_macd_hist and m.eth_macd_hist < -20:
                    # MACD still declining, wait
                    return None
        
        # BTC should not be dumping (RSI 4h > 40)
        if not isnan(m.btc_rsi_4h) and m.btc_rsi_4h < 40:
            return None
        
        # Calculate loan amount (more conservative for additional loans)
        if current_cycle_info and len(current_cycle_info.get("loans", [])) > 0:
            # For additional loans, use smaller amount (30% instead of 50%)
            loan_amount = max_loan * 0.3
            loan_amount = min(loan_amount, 15_000)  # Max 15k for additional loans
        else:
            # First loan in cycle: 50% of available
            loan_amount = max_loan * 0.5
            loan_amount = min(loan_amount, 20_000)  # Max 20k per trade
        
        self.last_eth_macd_hist = m.eth_macd_hist
        self.last_btc_macd_hist = m.btc_macd_hist
        
        return {
            "action": "BUY_ETH_WITH_LOAN",
            "loan_amount": round(loan_amount, 2),
            "reason": {
                "eth_price": m.eth_price,
                "eth_rsi_4h": m.eth_rsi_4h,
                "eth_macd_hist": m.eth_macd_hist,
                "buy_zone": (buy_low, buy_high),
            },
        }

    def should_sell_to_repay(self, m: MarketState, p: PositionState, current_cycle_info: Optional[Dict] = None) -> Optional[Dict]:
        """Sell ETH to repay loan when profitable or stop loss triggered"""
        if p.loan_usdt <= 0:
            return None
        
        # STOP LOSS: Check if price dropped too much from average buy price
        if current_cycle_info:
            cycle_avg_price = current_cycle_info.get("avg_buy_price", 0.0)
            if cycle_avg_price > 0:
                price_change_pct = ((m.eth_price - cycle_avg_price) / cycle_avg_price) * 100
                # Stop loss: if price dropped more than stop_loss_pct, sell immediately
                if price_change_pct <= self.stop_loss_pct:
                    print(f"[RISK] Stop loss triggered: price dropped {price_change_pct:.2f}% from avg buy price {cycle_avg_price:.2f}")
                    # Calculate how much ETH to sell to repay loan + buffer
                    buffer = 1.02  # 2% buffer for fees
                    eth_to_sell = (p.loan_usdt * buffer) / m.eth_price
                    # Don't sell more than 50% of position in stop loss
                    max_sell = p.spot_eth * 0.5
                    eth_to_sell = min(eth_to_sell, max_sell)
                    
                    if eth_to_sell > 0:
                        return {
                            "action": "SELL_ETH_REPAY_LOAN",
                            "eth_amount": round(eth_to_sell, 4),
                            "reason": {
                                "eth_price": m.eth_price,
                                "eth_rsi_4h": m.eth_rsi_4h,
                                "eth_macd_hist": m.eth_macd_hist,
                                "loan_usdt": p.loan_usdt,
                                "stop_loss": True,
                                "price_change_pct": price_change_pct,
                                "avg_buy_price": cycle_avg_price,
                            },
                        }
        
        sell_low, sell_high = self.conf.get("sell_zone", (0, 0))
        if sell_low <= 0 or sell_high <= 0:
            return None
        
        # Price should be in sell zone
        if not (sell_low <= m.eth_price <= sell_high):
            return None
        
        # RSI conditions: ETH RSI 4h should be high (overbought)
        rsi_high = self.conf.get("eth_rsi_4h_high", 70)
        if isnan(m.eth_rsi_4h) or m.eth_rsi_4h < rsi_high - 10:  # Not overbought enough
            return None
        
        # MACD: ETH MACD hist should be positive and potentially turning
        if not isnan(m.eth_macd_hist):
            if m.eth_macd_hist < 0:  # Still negative, wait
                return None
            # Prefer when MACD is turning from positive to negative
            if self.last_eth_macd_hist is not None:
                if m.eth_macd_hist > self.last_eth_macd_hist and m.eth_macd_hist > 50:
                    # MACD still rising strongly, might wait
                    pass
        
        # Calculate how much ETH to sell to repay loan + buffer
        buffer = 1.02  # 2% buffer for fees
        eth_to_sell = (p.loan_usdt * buffer) / m.eth_price
        
        # Don't sell more than 30% of position
        max_sell = p.spot_eth * 0.3
        eth_to_sell = min(eth_to_sell, max_sell)
        
        if eth_to_sell <= 0:
            return None
        
        self.last_eth_macd_hist = m.eth_macd_hist
        self.last_btc_macd_hist = m.btc_macd_hist
        
        return {
            "action": "SELL_ETH_REPAY_LOAN",
            "eth_amount": round(eth_to_sell, 4),
            "reason": {
                "eth_price": m.eth_price,
                "eth_rsi_4h": m.eth_rsi_4h,
                "eth_macd_hist": m.eth_macd_hist,
                "loan_usdt": p.loan_usdt,
                "sell_zone": (sell_low, sell_high),
            },
        }

    def should_take_profit(self, m: MarketState, p: PositionState) -> Optional[Dict]:
        """Take profit on some ETH when price is high (no loan)"""
        if p.loan_usdt > 0:
            return None  # Focus on repaying loan first
        
        sell_low, sell_high = self.conf.get("sell_zone", (0, 0))
        if sell_low <= 0 or sell_high <= 0:
            return None
        
        # Price should be in sell zone
        if not (sell_low <= m.eth_price <= sell_high):
            return None
        
        # Very high RSI (both 4h and 1d)
        rsi_high_4h = self.conf.get("eth_rsi_4h_high", 70)
        rsi_high_1d = self.conf.get("eth_rsi_1d_high", 70)
        
        if isnan(m.eth_rsi_4h) or m.eth_rsi_4h < rsi_high_4h:
            return None
        if isnan(m.eth_rsi_1d) or m.eth_rsi_1d < rsi_high_1d:
            return None
        
        # MACD very positive
        if isnan(m.eth_macd_hist) or m.eth_macd_hist < 30:
            return None
        
        # Sell 10-15% of position
        sell_pct = 0.12  # 12%
        eth_to_sell = p.spot_eth * sell_pct
        
        if eth_to_sell < 1.0:  # Minimum 1 ETH
            return None
        
        self.last_eth_macd_hist = m.eth_macd_hist
        self.last_btc_macd_hist = m.btc_macd_hist
        
        return {
            "action": "TAKE_PROFIT_SELL",
            "eth_amount": round(eth_to_sell, 4),
            "reason": {
                "eth_price": m.eth_price,
                "eth_rsi_4h": m.eth_rsi_4h,
                "eth_rsi_1d": m.eth_rsi_1d,
                "eth_macd_hist": m.eth_macd_hist,
            },
        }

    def should_buyback(self, m: MarketState, p: PositionState) -> Optional[Dict]:
        """Buy back ETH after taking profit"""
        # This would require tracking previous sells
        # For simplicity, we'll skip this for now
        return None

#############################
# ETH Rotation Bot
#############################

class EthRotationBot:
    def __init__(self):
        # Use Binance for better historical data, Bybit for real-time
        self.client = BinanceClient()  # Use Binance for market data
        self.bybit_client = BybitClient(BYBIT_API_KEY, BYBIT_API_SECRET)  # Keep Bybit for real-time if needed
        self.conf_builder = DynamicConfigBuilder(self.client)
        self.conf = self.conf_builder.build()
        self.strategy = EthRotationStrategy(self.conf)
        
        self.position = PositionState(
            spot_eth=INITIAL_ETH,
            loan_usdt=0.0,
            avg_entry_price=INITIAL_ENTRY_PRICE,
        )
        
        self.last_conf_update = 0
        self.conf_update_interval = 4 * 60 * 60  # 4 hours

    def build_market_state(self) -> MarketState:
        eth_price = self.client.get_ticker("ETHUSDT")["last"]
        btc_price = self.client.get_ticker("BTCUSDT")["last"]
        
        # Get OHLCV data
        eth_4h = self.client.get_ohlcv("ETHUSDT", "4h", 100)
        eth_1d = self.client.get_ohlcv("ETHUSDT", "1d", 100)
        btc_4h = self.client.get_ohlcv("BTCUSDT", "4h", 100)
        btc_1d = self.client.get_ohlcv("BTCUSDT", "1d", 100)
        
        # Calculate indicators
        eth_closes_4h = [c["close"] for c in eth_4h] if eth_4h else []
        eth_closes_1d = [c["close"] for c in eth_1d] if eth_1d else []
        btc_closes_4h = [c["close"] for c in btc_4h] if btc_4h else []
        btc_closes_1d = [c["close"] for c in btc_1d] if btc_1d else []
        
        eth_rsi_4h = calc_rsi(eth_closes_4h) if eth_closes_4h else float("nan")
        eth_rsi_1d = calc_rsi(eth_closes_1d) if eth_closes_1d else float("nan")
        btc_rsi_4h = calc_rsi(btc_closes_4h) if btc_closes_4h else float("nan")
        btc_rsi_1d = calc_rsi(btc_closes_1d) if btc_closes_1d else float("nan")
        
        _, _, eth_macd_hist = calc_macd(eth_closes_4h) if eth_closes_4h else (float("nan"),)*3
        _, _, btc_macd_hist = calc_macd(btc_closes_4h) if btc_closes_4h else (float("nan"),)*3
        
        return MarketState(
            eth_price=eth_price,
            btc_price=btc_price,
            eth_rsi_4h=eth_rsi_4h,
            eth_rsi_1d=eth_rsi_1d,
            btc_rsi_4h=btc_rsi_4h,
            btc_rsi_1d=btc_rsi_1d,
            eth_macd_hist=eth_macd_hist,
            btc_macd_hist=btc_macd_hist,
        )

    def maybe_update_conf(self):
        global LAST_CONFIG
        now = time.time()
        if now - self.last_conf_update > self.conf_update_interval:
            self.conf = self.conf_builder.build()
            self.strategy.update_conf(self.conf)
            self.last_conf_update = now
            LAST_CONFIG = self.conf.copy()
            print(f"[CONF] Updated: {self.conf}")

    def handle_signal(self, signal: Dict[str, Any], m: MarketState):
        global RECENT_SIGNALS
        
        action = signal["action"]
        payload = {
            "action": action,
            "signal": signal,
            "market": asdict(m),
            "position": asdict(self.position),
            "ts": int(time.time()),
        }
        
        RECENT_SIGNALS.append(payload)
        if len(RECENT_SIGNALS) > MAX_SIGNALS:
            RECENT_SIGNALS = RECENT_SIGNALS[-MAX_SIGNALS:]
        
        # Update position (virtual)
        if action == "BUY_ETH_WITH_LOAN":
            loan = signal["loan_amount"]
            self.position.loan_usdt += loan
            bought_eth = loan / m.eth_price
            self.position.spot_eth += bought_eth
            # Update avg entry (weighted)
            total_value = (self.position.spot_eth - bought_eth) * self.position.avg_entry_price + loan
            self.position.avg_entry_price = total_value / self.position.spot_eth if self.position.spot_eth > 0 else self.position.avg_entry_price
            
        elif action == "SELL_ETH_REPAY_LOAN":
            eth_sold = signal["eth_amount"]
            self.position.spot_eth = max(0.0, self.position.spot_eth - eth_sold)
            self.position.loan_usdt = 0.0
            
        elif action == "TAKE_PROFIT_SELL":
            eth_sold = signal["eth_amount"]
            self.position.spot_eth = max(0.0, self.position.spot_eth - eth_sold)
        
        # Notifications
        title = f"ETH Rotation: {action}"
        msg = (
            f"{action}\n"
            f"ETH: ${m.eth_price:.2f}\n"
            f"Position: {self.position.spot_eth:.4f} ETH\n"
            f"Loan: ${self.position.loan_usdt:.2f}\n"
            f"RSI 4h: {m.eth_rsi_4h:.2f}\n"
            f"MACD Hist: {m.eth_macd_hist:.2f}\n"
            f"{signal}"
        )
        pushover_notify(title, msg)
        telegram_notify(msg)
        
        # Log to Supabase
        supabase_log_signal(action, asdict(m), signal)
        
        print(f"[SIGNAL] {payload}")

    def run_step(self):
        global LAST_MARKET_STATE
        
        self.maybe_update_conf()
        m = self.build_market_state()
        LAST_MARKET_STATE = asdict(m)
        
        # Check signals in priority order (for real bot, we don't track cycles, so pass None)
        signal = self.strategy.should_sell_to_repay(m, self.position, None)
        if not signal:
            signal = self.strategy.should_take_profit(m, self.position)
        if not signal:
            signal = self.strategy.should_buy_with_loan(m, self.position, None)
        
        if signal:
            self.handle_signal(signal, m)

#############################
# Backtesting Module
#############################

class RotationBacktester:
    def __init__(self, client=None):
        # Use Binance by default for better historical data access
        self.client = client or BinanceClient()
        self.use_binance = isinstance(self.client, BinanceClient)
        if self.use_binance:
            print(f"[BACKTEST] Using Binance API for historical data (better access to historical data)")
        elif hasattr(self.client, 'use_auth') and self.client.use_auth:
            print(f"[BACKTEST] Using authenticated Bybit API requests")
        else:
            print(f"[BACKTEST] Using public Bybit API (no authentication). Set BYBIT_API_KEY and BYBIT_API_SECRET for better access.")

    def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
        
        print(f"[BACKTEST] Fetching {symbol} {interval} data from {start_date} to {end_date} (ts: {start_ts} to {end_ts})")
        
        # Use Binance if available (better historical data)
        if self.use_binance:
            return self.client.get_ohlcv_range(symbol, interval, start_ts, end_ts)
        
        # Fallback to Bybit
        all_candles = []
        current_start = start_ts * 1000
        end_ms = end_ts * 1000
        
        interval_ms = 4 * 60 * 60 * 1000 if interval == "4h" else 24 * 60 * 60 * 1000
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while current_start < end_ms and iteration < max_iterations:
            iteration += 1
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": self.client._map_interval(interval),
                "start": current_start,
                "end": end_ms,
                "limit": 200,
            }
            # Use authenticated request if API key is available (may have better rate limits and access)
            use_auth = getattr(self.client, 'use_auth', False)
            data = self.client._get("/v5/market/kline", params, use_auth=use_auth)
            if not data or "result" not in data or not data["result"].get("list"):
                print(f"[BACKTEST] No more data for {symbol} {interval} at iteration {iteration}")
                break
            
            batch = []
            for item in reversed(data["result"]["list"]):  # Reverse to get chronological order
                ts_ms = int(item[0])
                if start_ts * 1000 <= ts_ms < end_ms:
                    batch.append({
                        "timestamp": ts_ms // 1000,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    })
            
            if not batch:
                print(f"[BACKTEST] Empty batch for {symbol} {interval} at iteration {iteration}")
                break
            
            # Remove duplicates based on timestamp
            existing_timestamps = {c["timestamp"] for c in all_candles}
            new_batch = [c for c in batch if c["timestamp"] not in existing_timestamps]
            
            if not new_batch:
                # All duplicates, move forward
                current_start = max(c["timestamp"] * 1000 for c in batch) + interval_ms
                continue
            
            all_candles.extend(new_batch)
            
            # Update current_start to the latest timestamp + interval
            latest_ts = max(c["timestamp"] for c in new_batch)
            current_start = latest_ts * 1000 + interval_ms
            
            print(f"[BACKTEST] Fetched {len(new_batch)} new candles for {symbol} {interval} (total: {len(all_candles)}, latest: {datetime.fromtimestamp(latest_ts).strftime('%Y-%m-%d %H:%M')})")
            
            if len(batch) < 200:
                # Last batch
                break
        
        all_candles.sort(key=lambda x: x["timestamp"])
        print(f"[BACKTEST] Retrieved {len(all_candles)} unique candles for {symbol} {interval} (range: {datetime.fromtimestamp(all_candles[0]['timestamp']).strftime('%Y-%m-%d') if all_candles else 'N/A'} to {datetime.fromtimestamp(all_candles[-1]['timestamp']).strftime('%Y-%m-%d') if all_candles else 'N/A'})")
        return all_candles

    def run_backtest(self, start_date: str, end_date: str, use_realtime_config: bool = False, config_update_candles: int = 24) -> Dict[str, Any]:
        config_mode = "REAL-TIME" if use_realtime_config else "HISTORICAL"
        print(f"[BACKTEST] Starting from {start_date} to {end_date} (Config: {config_mode}, Update every {config_update_candles} candles)")
        
        # Validate dates are in the past
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        now = datetime.now()
        
        if start_dt > now or end_dt > now:
            return {"error": "Backtest dates must be in the past"}
        
        if start_dt >= end_dt:
            return {"error": "Start date must be before end date"}
        
        eth_4h = self.get_historical_data("ETHUSDT", "4h", start_date, end_date)
        eth_1d = self.get_historical_data("ETHUSDT", "1d", start_date, end_date)
        btc_4h = self.get_historical_data("BTCUSDT", "4h", start_date, end_date)
        btc_1d = self.get_historical_data("BTCUSDT", "1d", start_date, end_date)
        
        print(f"[BACKTEST] Data summary: ETH 4h: {len(eth_4h)}, ETH 1d: {len(eth_1d)}, BTC 4h: {len(btc_4h)}, BTC 1d: {len(btc_1d)}")
        
        if len(eth_4h) < 100:
            return {"error": f"Not enough data. Only got {len(eth_4h)} candles for ETHUSDT 4h. Need at least 100."}
        
        position = PositionState(
            spot_eth=INITIAL_ETH,
            loan_usdt=0.0,
            avg_entry_price=INITIAL_ENTRY_PRICE,
        )
        
        # Build config based on historical data in backtest period, not current data
        # Calculate dynamic config from the backtest data itself
        def build_config_from_backtest_data(eth_4h_data, eth_1d_data, current_index):
            """Build config from backtest historical data up to current point"""
            if current_index < 50:
                # Not enough data yet, use default
                return {
                    "eth_rsi_4h_low": 30,
                    "eth_rsi_4h_high": 70,
                    "eth_rsi_1d_low": 30,
                    "eth_rsi_1d_high": 70,
                    "buy_zone": (0, 0),
                    "sell_zone": (0, 0),
                    "price_range": (0, 0),
                    "current_price_position": 0.5,
                }
            
            # Use data up to current point
            eth_closes_4h = [c["close"] for c in eth_4h_data[:current_index+1]]
            eth_closes_1d = [c["close"] for c in eth_1d_data if c["timestamp"] <= eth_4h_data[current_index]["timestamp"]]
            
            # Calculate RSI series for dynamic thresholds
            eth_rsi_4h_series = []
            eth_rsi_1d_series = []
            for i in range(50, len(eth_closes_4h)):
                rsi = calc_rsi(eth_closes_4h[:i+1])
                if not isnan(rsi):
                    eth_rsi_4h_series.append(rsi)
            for i in range(50, len(eth_closes_1d)):
                rsi = calc_rsi(eth_closes_1d[:i+1])
                if not isnan(rsi):
                    eth_rsi_1d_series.append(rsi)
            
            # Dynamic RSI thresholds (20th and 80th percentile)
            eth_rsi_4h_low = np.percentile(eth_rsi_4h_series, 20) if eth_rsi_4h_series else 30
            eth_rsi_4h_high = np.percentile(eth_rsi_4h_series, 80) if eth_rsi_4h_series else 70
            eth_rsi_1d_low = np.percentile(eth_rsi_1d_series, 20) if eth_rsi_1d_series else 30
            eth_rsi_1d_high = np.percentile(eth_rsi_1d_series, 80) if eth_rsi_1d_series else 70
            
            # Calculate buy/sell zones based on recent price action (last 60 candles or available)
            lookback = min(60, len(eth_closes_4h))
            recent_prices = eth_closes_4h[-lookback:]
            price_low = min(recent_prices)
            price_high = max(recent_prices)
            price_range = price_high - price_low
            
            # Buy zone: Lower 30% of range
            buy_zone_low = price_low
            buy_zone_high = price_low + price_range * 0.3
            
            # Sell zone: Upper 30% of range
            sell_zone_low = price_high - price_range * 0.3
            sell_zone_high = price_high
            
            # Current price position in range
            current_price = eth_closes_4h[-1]
            price_position = (current_price - price_low) / price_range if price_range > 0 else 0.5
            
            return {
                "eth_rsi_4h_low": float(eth_rsi_4h_low),
                "eth_rsi_4h_high": float(eth_rsi_4h_high),
                "eth_rsi_1d_low": float(eth_rsi_1d_low),
                "eth_rsi_1d_high": float(eth_rsi_1d_high),
                "buy_zone": (float(buy_zone_low), float(buy_zone_high)),
                "sell_zone": (float(sell_zone_low), float(sell_zone_high)),
                "price_range": (float(price_low), float(price_high)),
                "current_price_position": float(price_position),
            }
        
        # Initial config: use real-time config or backtest data config
        if use_realtime_config:
            # Use real-time config from DynamicConfigBuilder
            print(f"[BACKTEST] Using REAL-TIME config (from current market data)")
            binance_client = BinanceClient()
            conf_builder = DynamicConfigBuilder(binance_client)
            conf = conf_builder.build()
            print(f"[BACKTEST] Real-time config: buy_zone={conf['buy_zone']}, sell_zone={conf['sell_zone']}, RSI_4h_low={conf['eth_rsi_4h_low']:.1f}, RSI_4h_high={conf['eth_rsi_4h_high']:.1f}")
        else:
            # Use config from backtest historical data
            print(f"[BACKTEST] Using HISTORICAL config (from backtest data)")
            conf = build_config_from_backtest_data(eth_4h, eth_1d, 100)
        strategy = EthRotationStrategy(conf)
        
        signals = []
        equity_curve = []
        price_history = []  # Store price history for chart
        cycles = []  # Track buy-sell cycles
        current_cycle = None  # Current active cycle
        
        initial_value = INITIAL_ETH * eth_4h[0]["close"]
        equity_curve.append(initial_value)
        
        max_equity = initial_value
        max_drawdown = 0.0
        
        for i in range(100, len(eth_4h)):
            candle = eth_4h[i]
            candle_ts = candle["timestamp"]
            
            # Get corresponding prices
            btc_price = btc_4h[i]["close"] if i < len(btc_4h) else btc_4h[-1]["close"] if btc_4h else 0
            
            # Calculate indicators
            eth_closes_4h = [c["close"] for c in eth_4h[:i+1]]
            eth_closes_1d = [c["close"] for c in eth_1d if c["timestamp"] <= candle_ts]
            btc_closes_4h = [c["close"] for c in btc_4h[:i+1]] if i < len(btc_4h) else []
            btc_closes_1d = [c["close"] for c in btc_1d if c["timestamp"] <= candle_ts] if btc_1d else []
            
            eth_rsi_4h = calc_rsi(eth_closes_4h[-80:]) if len(eth_closes_4h) >= 80 else float("nan")
            eth_rsi_1d = calc_rsi(eth_closes_1d) if len(eth_closes_1d) >= 14 else float("nan")
            btc_rsi_4h = calc_rsi(btc_closes_4h[-80:]) if len(btc_closes_4h) >= 80 else float("nan")
            btc_rsi_1d = calc_rsi(btc_closes_1d) if len(btc_closes_1d) >= 14 else float("nan")
            
            _, _, eth_macd_hist = calc_macd(eth_closes_4h[-80:]) if len(eth_closes_4h) >= 80 else (float("nan"),)*3
            _, _, btc_macd_hist = calc_macd(btc_closes_4h[-80:]) if len(btc_closes_4h) >= 80 else (float("nan"),)*3
            
            # Update config periodically
            if i % config_update_candles == 0 or i == 100:  # Update every N candles or at start
                try:
                    if use_realtime_config:
                        # Keep using real-time config (don't update during backtest)
                        # Config was already set at the beginning
                        pass
                    else:
                        # Update config based on backtest data up to current point
                        conf = build_config_from_backtest_data(eth_4h, eth_1d, i)
                        strategy.update_conf(conf)
                        print(f"[BACKTEST] Config updated at candle {i}: buy_zone={conf['buy_zone']}, sell_zone={conf['sell_zone']}, RSI_4h_low={conf['eth_rsi_4h_low']:.1f}, RSI_4h_high={conf['eth_rsi_4h_high']:.1f}")
                except Exception as e:
                    print(f"[BACKTEST] Error updating config: {e}")
                    pass
            
            market_state = MarketState(
                eth_price=candle["close"],
                btc_price=btc_price,
                eth_rsi_4h=eth_rsi_4h,
                eth_rsi_1d=eth_rsi_1d,
                btc_rsi_4h=btc_rsi_4h,
                btc_rsi_1d=btc_rsi_1d,
                eth_macd_hist=eth_macd_hist,
                btc_macd_hist=btc_macd_hist,
            )
            
            # Check signals (pass current cycle info for risk management)
            signal = strategy.should_sell_to_repay(market_state, position, current_cycle)
            if not signal:
                signal = strategy.should_take_profit(market_state, position)
            if not signal:
                signal = strategy.should_buy_with_loan(market_state, position, current_cycle)
            
            if signal:
                action = signal["action"]
                
                if action == "BUY_ETH_WITH_LOAN":
                    loan = signal["loan_amount"]
                    position.loan_usdt += loan
                    bought_eth = loan / market_state.eth_price
                    position.spot_eth += bought_eth
                    total_value = (position.spot_eth - bought_eth) * position.avg_entry_price + loan
                    position.avg_entry_price = total_value / position.spot_eth if position.spot_eth > 0 else position.avg_entry_price
                    
                    # Start new cycle or add to existing cycle
                    if current_cycle is None:
                        current_cycle = {
                            "start_timestamp": candle_ts,
                            "start_candle": i - 100,
                            "loans": [],
                            "total_loan": 0.0,
                            "total_eth_bought": 0.0,
                            "avg_buy_price": 0.0,
                            "sell_timestamp": None,
                            "sell_candle": None,
                            "eth_sold": 0.0,
                            "sell_price": 0.0,
                            "revenue": 0.0,
                            "eth_gained": 0.0,
                            "profit_usdt": 0.0,
                        }
                    
                    # Add this buy to cycle
                    current_cycle["loans"].append({
                        "loan_amount": loan,
                        "eth_bought": bought_eth,
                        "buy_price": market_state.eth_price,
                    })
                    current_cycle["total_loan"] += loan
                    current_cycle["total_eth_bought"] += bought_eth
                    current_cycle["avg_buy_price"] = current_cycle["total_loan"] / current_cycle["total_eth_bought"] if current_cycle["total_eth_bought"] > 0 else 0
                    print(f"[BACKTEST] BUY_ETH_WITH_LOAN: ${loan:.2f}, {bought_eth:.4f} ETH at ${market_state.eth_price:.2f} (Cycle has {len(current_cycle['loans'])} loans)")
                    
                elif action == "SELL_ETH_REPAY_LOAN":
                    eth_sold = signal["eth_amount"]
                    position.spot_eth = max(0.0, position.spot_eth - eth_sold)
                    position.loan_usdt = 0.0
                    
                    # Complete cycle if active
                    if current_cycle:
                        current_cycle["sell_timestamp"] = candle_ts
                        current_cycle["sell_candle"] = i - 100
                        current_cycle["eth_sold"] = eth_sold
                        current_cycle["sell_price"] = market_state.eth_price
                        current_cycle["revenue"] = eth_sold * market_state.eth_price
                        current_cycle["profit_usdt"] = current_cycle["revenue"] - current_cycle["total_loan"]
                        # Calculate ETH gained (remaining after repaying loan)
                        current_cycle["eth_gained"] = current_cycle["total_eth_bought"] - eth_sold
                        # Check if stop loss was triggered
                        stop_loss = signal.get("reason", {}).get("stop_loss", False)
                        cycles.append(current_cycle)
                        stop_loss_msg = " [STOP LOSS]" if stop_loss else ""
                        print(f"[BACKTEST] Cycle completed{stop_loss_msg}: {len(current_cycle.get('loans', []))} loans, {current_cycle['total_eth_bought']:.4f} ETH bought, {eth_sold:.4f} ETH sold, profit: ${current_cycle['profit_usdt']:.2f}, ETH gained: {current_cycle['eth_gained']:.4f}")
                        current_cycle = None
                    else:
                        # SELL without BUY - this shouldn't happen but log it
                        print(f"[BACKTEST] WARNING: SELL_ETH_REPAY_LOAN without active cycle at candle {i}")
                    
                elif action == "TAKE_PROFIT_SELL":
                    eth_sold = signal["eth_amount"]
                    position.spot_eth = max(0.0, position.spot_eth - eth_sold)
                    # Take profit doesn't close a cycle, just reduces position
                    # But if we have an active cycle, we might want to track this differently
                    # For now, we'll keep it as is - take profit is separate from loan cycles
                
                signals.append({
                    "timestamp": candle_ts,
                    "action": action,
                    "eth_price": market_state.eth_price,
                    "signal": signal,
                    "candle_index": i - 100,  # Index in equity curve
                })
            
            current_value = position.spot_eth * market_state.eth_price - position.loan_usdt
            equity_curve.append(current_value)
            
            # Store price history for chart
            price_history.append({
                "timestamp": candle_ts,
                "price": market_state.eth_price,
                "candle_index": i - 100,  # Index in equity curve
            })
            
            if current_value > max_equity:
                max_equity = current_value
            
            drawdown = ((max_equity - current_value) / max_equity) * 100 if max_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Close any open cycle at the end
        if current_cycle:
            # Mark as incomplete
            current_cycle["incomplete"] = True
            cycles.append(current_cycle)
        
        final_value = equity_curve[-1] if equity_curve else initial_value
        return_pct = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        signal_counts = {}
        for s in signals:
            action = s["action"]
            signal_counts[action] = signal_counts.get(action, 0) + 1
        
        # Debug: Count signal types
        buy_signals = [s for s in signals if s["action"] == "BUY_ETH_WITH_LOAN"]
        sell_signals = [s for s in signals if s["action"] == "SELL_ETH_REPAY_LOAN"]
        profit_signals = [s for s in signals if s["action"] == "TAKE_PROFIT_SELL"]
        print(f"[BACKTEST] Signals: {len(signals)} total ({len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(profit_signals)} PROFIT)")
        print(f"[BACKTEST] Cycles: {len(cycles)} ({sum(1 for c in cycles if not c.get('incomplete', False))} completed, {sum(1 for c in cycles if c.get('incomplete', False))} incomplete)")
        
        # Sample price history to match equity curve sampling (max 2000 points for better detail)
        max_points = 2000
        equity_sample_step = max(1, len(equity_curve) // max_points)
        sampled_price_history = price_history[::equity_sample_step]
        sampled_equity_curve = equity_curve[::equity_sample_step]
        print(f"[BACKTEST] Sampled {len(sampled_equity_curve)} points from {len(equity_curve)} total (step: {equity_sample_step})")
        
        # Map signals to sampled indices
        signal_indices = {}
        for signal in signals:
            # Find which sampled candle this signal belongs to
            signal_candle_idx = signal.get("candle_index", 0)
            sampled_idx = signal_candle_idx // equity_sample_step
            if sampled_idx < len(sampled_equity_curve):
                if sampled_idx not in signal_indices:
                    signal_indices[sampled_idx] = []
                # Store signal with its original index for reference
                signal_with_idx = signal.copy()
                signal_with_idx["sampled_index"] = sampled_idx
                signal_indices[sampled_idx].append(signal_with_idx)
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_value": initial_value,
            "final_value": final_value,
            "return_pct": return_pct,
            "max_drawdown": max_drawdown,
            "total_signals": len(signals),
            "signal_breakdown": signal_counts,
            "final_eth": position.spot_eth,
            "final_loan": position.loan_usdt,
            "equity_curve": sampled_equity_curve,
            "price_history": sampled_price_history,
            "signals": signals,  # Return all signals, not just last 20
            "signal_indices": signal_indices,  # Map of index -> signals
            "cycles": cycles,  # Buy-sell cycles
        }

#############################
# FastAPI App
#############################

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

app = FastAPI(title="ETH Rotation Bot")

bot = EthRotationBot()

def bot_loop():
    while True:
        try:
            bot.run_step()
        except Exception as e:
            print(f"[BOT ERROR] {e}")
        time.sleep(300)  # Run every 5 minutes

@app.on_event("startup")
def on_startup():
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    print("[BOT] Started rotation bot")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    m = LAST_MARKET_STATE or {}
    c = LAST_CONFIG or {}
    pos = bot.position
    
    def fmt(v, d=2):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) and not isnan(v) else "N/A"
    
    # Prepare chart data
    signals_data = []
    for s in RECENT_SIGNALS[-100:]:
        signals_data.append({
            "time": s["ts"],
            "action": s["action"],
            "price": s["market"].get("eth_price", 0)
        })
    
    # Get historical price data for chart (even without signals)
    price_history = []
    try:
        eth_4h = bot.client.get_ohlcv("ETHUSDT", "4h", 100)
        if eth_4h and len(eth_4h) > 0:
            for candle in eth_4h[-50:]:  # Last 50 candles
                price_history.append({
                    "time": candle["timestamp"],
                    "price": candle["close"],
                    "high": candle["high"],
                    "low": candle["low"],
                })
    except Exception as e:
        print(f"[DASHBOARD] Error fetching price history: {e}")
        # Fallback: use current price if available
        if m.get("eth_price"):
            price_history.append({
                "time": int(time.time()),
                "price": m.get("eth_price"),
                "high": m.get("eth_price"),
                "low": m.get("eth_price"),
            })
    
    buy_zone = c.get("buy_zone", (0, 0))
    sell_zone = c.get("sell_zone", (0, 0))
    price_range = c.get("price_range", (0, 0))
    
    current_value = pos.spot_eth * m.get("eth_price", 0) - pos.loan_usdt
    pnl_pct = ((m.get("eth_price", 0) - pos.avg_entry_price) / pos.avg_entry_price * 100) if pos.avg_entry_price > 0 else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ETH Rotation Bot</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap" rel="stylesheet">
        <style>
            /* Pixel chart styles - keep for charts only */
            @keyframes scanline {{
                0% {{ transform: translateY(0); }}
                100% {{ transform: translateY(100vh); }}
            }}
            @keyframes flicker {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.98; }}
            }}
            
            .pixel-chart-container {{
                position: relative;
                background: #000000;
                border: 3px solid #00ff00;
                padding: 10px;
                box-shadow: inset 0 0 20px rgba(0,255,0,0.2);
            }}
            
            .pixel-chart-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(transparent 50%, rgba(0,255,0,0.03) 50%);
                background-size: 100% 4px;
                pointer-events: none;
                animation: scanline 8s linear infinite;
                z-index: 1;
            }}
            
            /* Modern flat minimalist styles */
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f5f7fa;
                color: #2d3748;
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{ 
                max-width: 1400px; 
                margin: 0 auto; 
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 40px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
            }}
            
            .header h1 {{
                color: #ffffff;
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 10px;
                letter-spacing: -0.5px;
            }}
            
            .header p {{
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1em;
                font-weight: 300;
            }}
            
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 24px; 
                margin-bottom: 24px; 
            }}
            
            .card {{
                background: #ffffff;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
            }}
            
            .card h2 {{
                color: #1a202c;
                font-size: 1.25em;
                font-weight: 600;
                margin-bottom: 20px;
                padding-bottom: 12px;
                border-bottom: 2px solid #e2e8f0;
            }}
            
            .metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 0;
                border-bottom: 1px solid #e2e8f0;
            }}
            
            .metric:last-child {{
                border-bottom: none;
            }}
            
            .metric span:first-child {{
                color: #718096;
                font-size: 0.95em;
            }}
            
            .metric-value {{
                font-weight: 600;
                font-size: 1.1em;
                color: #2d3748;
            }}
            
            .positive {{ 
                color: #48bb78; 
            }}
            
            .negative {{ 
                color: #f56565; 
            }}
            
            .chart-container {{ 
                position: relative; 
                height: 400px; 
                margin-top: 20px;
            }}
            
            .zone-buy {{
                background: #f0fff4;
                border-left: 4px solid #48bb78;
                padding: 16px;
                margin: 12px 0;
                border-radius: 8px;
            }}
            
            .zone-sell {{
                background: #fff5f5;
                border-left: 4px solid #f56565;
                padding: 16px;
                margin: 12px 0;
                border-radius: 8px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            
            th {{
                background: #f7fafc;
                color: #4a5568;
                font-weight: 600;
                font-size: 0.875em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            td {{
                color: #2d3748;
                font-size: 0.95em;
            }}
            
            tr:hover {{
                background: #f7fafc;
            }}
            
            button {{
                font-family: inherit;
                font-size: 0.95em;
                font-weight: 600;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            
            button:first-of-type {{
                background: linear-gradient(135deg, #f6ad55 0%, #ed8936 100%);
                color: #ffffff;
            }}
            
            button:first-of-type:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(246, 173, 85, 0.4);
            }}
            
            button:last-of-type {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff;
            }}
            
            button:last-of-type:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            
            input[type="date"], input[type="number"] {{
                font-family: inherit;
                font-size: 1em;
                padding: 10px 14px;
                border: 2px solid #e2e8f0;
                background: #ffffff;
                color: #2d3748;
                width: 100%;
                border-radius: 8px;
                transition: border-color 0.2s;
            }}
            
            input[type="date"]:focus, input[type="number"]:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }}
            
            label {{
                color: #4a5568;
                font-size: 0.875em;
                font-weight: 600;
                display: block;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            small {{
                font-size: 0.875em;
                color: #718096;
            }}
            
            #backtestResults {{
                margin-top: 20px;
            }}
            
            #backtestResults > div {{
                background: #ffffff;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ETH ROTATION BOT</h1>
                <p>INCREASE ETH HOLDINGS THROUGH MARGIN ROTATION</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>POSITION</h2>
                    <div class="metric">
                        <span>ETH Holdings</span>
                        <span class="metric-value">{fmt(pos.spot_eth, 4)} ETH</span>
                    </div>
                    <div class="metric">
                        <span>Position Value</span>
                        <span class="metric-value">${fmt(current_value, 2)}</span>
                    </div>
                    <div class="metric">
                        <span>Loan USDT</span>
                        <span class="metric-value">${fmt(pos.loan_usdt, 2)}</span>
                    </div>
                    <div class="metric">
                        <span>Avg Entry</span>
                        <span class="metric-value">${fmt(pos.avg_entry_price, 2)}</span>
                    </div>
                    <div class="metric">
                        <span>P&L %</span>
                        <span class="metric-value {'positive' if pnl_pct >= 0 else 'negative'}">{fmt(pnl_pct, 2)}%</span>
                    </div>
                    <div class="metric">
                        <span>LTV</span>
                        <span class="metric-value">{fmt((pos.loan_usdt / (pos.spot_eth * m.get('eth_price', 1)) * 100) if m.get('eth_price', 0) > 0 else 0, 2)}%</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Market</h2>
                    <div class="metric">
                        <span>ETH Price</span>
                        <span class="metric-value">${fmt(m.get('eth_price'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>BTC Price</span>
                        <span class="metric-value">${fmt(m.get('btc_price'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>ETH RSI 4H</span>
                        <span class="metric-value">{fmt(m.get('eth_rsi_4h'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>ETH RSI 1D</span>
                        <span class="metric-value">{fmt(m.get('eth_rsi_1d'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>ETH MACD Hist</span>
                        <span class="metric-value">{fmt(m.get('eth_macd_hist'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>BTC MACD Hist</span>
                        <span class="metric-value">{fmt(m.get('btc_macd_hist'), 2)}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Dynamic Zones</h2>
                    <div class="zone-buy">
                        <strong style="color: #48bb78;">Buy Zone:</strong><br>
                        ${fmt(buy_zone[0], 2)} - ${fmt(buy_zone[1], 2)}
                    </div>
                    <div class="zone-sell">
                        <strong style="color: #f56565;">Sell Zone:</strong><br>
                        ${fmt(sell_zone[0], 2)} - ${fmt(sell_zone[1], 2)}
                    </div>
                    <div class="metric">
                        <span>Current Price Position</span>
                        <span class="metric-value">{fmt(c.get('current_price_position', 0) * 100, 1)}%</span>
                    </div>
                    <div class="metric">
                        <span>RSI 4H Low</span>
                        <span class="metric-value">{fmt(c.get('eth_rsi_4h_low'), 2)}</span>
                    </div>
                    <div class="metric">
                        <span>RSI 4H High</span>
                        <span class="metric-value">{fmt(c.get('eth_rsi_4h_high'), 2)}</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Price Chart with Zones</h2>
                <div class="chart-container pixel-chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Recent Signals</h2>
                <div style="margin-bottom: 15px; display: flex; align-items: center; gap: 12px;">
                    <button onclick="loadSignalsFromSupabase()" style="background: #4299e1; color: white;">
                        Load from Database
                    </button>
                    <span id="signalCount" style="color: #718096; font-size: 0.9em;">Showing {len(RECENT_SIGNALS)} signals</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Action</th>
                            <th>ETH Price</th>
                            <th>RSI 4H</th>
                            <th>MACD Hist</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody id="signalsTableBody">
                        {"".join([f"""
                        <tr>
                            <td>{time.strftime('%Y-%m-%d %H:%M', time.localtime(s['ts']))}</td>
                            <td><strong style="color: {'#48bb78' if 'BUY' in s['action'] else '#f56565' if 'SELL' in s['action'] else '#ed8936'}">{s['action']}</strong></td>
                            <td>${fmt(s['market'].get('eth_price'), 2)}</td>
                            <td>{fmt(s['market'].get('eth_rsi_4h'), 2)}</td>
                            <td>{fmt(s['market'].get('eth_macd_hist'), 2)}</td>
                            <td>
                                {f"Loan: ${fmt(s['signal'].get('loan_amount', 0), 2)}" if 'loan_amount' in s.get('signal', {}) else ''}
                                {f"ETH: {fmt(s['signal'].get('eth_amount', 0), 4)}" if 'eth_amount' in s.get('signal', {}) else ''}
                            </td>
                        </tr>
                        """ for s in reversed(RECENT_SIGNALS[-30:])]) or "<tr><td colspan='6'>No signals yet</td></tr>"}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>Backtest</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div>
                        <label>Start Date:</label>
                        <input type="date" id="backtestStart">
                    </div>
                    <div>
                        <label>End Date:</label>
                        <input type="date" id="backtestEnd">
                    </div>
                    <div id="configUpdateContainer" style="display: none;">
                        <label>Config Update (candles):</label>
                        <input type="number" id="configUpdateCandles" min="1" max="200" value="24">
                        <small>Update config every N candles (Historical mode only)</small>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-end; gap: 10px; margin-bottom: 10px;">
                    <button onclick="runBacktest(false); toggleConfigUpdateInput(true);">Run Backtest</button>
                    <button onclick="runBacktest(true); toggleConfigUpdateInput(false);">Backtest Real</button>
                </div>
                <div style="margin-top: 10px; padding: 16px; background: #edf2f7; border-radius: 8px; font-size: 0.9em; color: #4a5568;">
                    <strong style="color: #2d3748;">Note:</strong> "Run Backtest" uses config from historical data (updates periodically). "Backtest Real" uses current real-time config (same as live bot).
                </div>
                <div id="backtestResults"></div>
                <div class="chart-container pixel-chart-container" id="backtestChartContainer" style="display: none;">
                    <canvas id="backtestChart"></canvas>
                </div>
            </div>
        </div>
        
        <script>
            const signalsData = {json.dumps(signals_data)};
            const priceHistory = {json.dumps(price_history)};
            const buyZone = {json.dumps(buy_zone)};
            const sellZone = {json.dumps(sell_zone)};
            const priceRange = {json.dumps(price_range)};
            
            // Use price history if available, otherwise use signals
            let chartData = priceHistory && priceHistory.length > 0 ? priceHistory : signalsData;
            let chartLabels = [];
            let chartPrices = [];
            
            if (chartData && chartData.length > 0) {{
                chartLabels = chartData.map(d => new Date(d.time * 1000).toLocaleString());
                chartPrices = chartData.map(d => d.price || d.close || 0);
            }}
            
            // Prepare signal markers - separate datasets for each signal type
            const buySignals = [];
            const sellSignals = [];
            const profitSignals = [];
            
            if (signalsData && signalsData.length > 0 && chartLabels.length > 0) {{
                signalsData.forEach(signal => {{
                    const signalTime = new Date(signal.time * 1000);
                    // Find closest price point
                    let closestIdx = 0;
                    let minDiff = Infinity;
                    chartLabels.forEach((label, idx) => {{
                        const labelTime = new Date(label);
                        const diff = Math.abs(signalTime - labelTime);
                        if (diff < minDiff) {{
                            minDiff = diff;
                            closestIdx = idx;
                        }}
                    }});
                    
                    // Only add if within 4 hours
                    if (minDiff < 1000 * 60 * 60 * 4) {{
                        const signalPoint = {{
                            x: closestIdx,
                            y: signal.price,
                            action: signal.action,
                            time: signal.time
                        }};
                        
                        if (signal.action === 'BUY_ETH_WITH_LOAN') {{
                            buySignals.push(signalPoint);
                        }} else if (signal.action === 'SELL_ETH_REPAY_LOAN') {{
                            sellSignals.push(signalPoint);
                        }} else if (signal.action === 'TAKE_PROFIT_SELL') {{
                            profitSignals.push(signalPoint);
                        }}
                    }}
                }});
            }}
            
            const ctx = document.getElementById('priceChart');
            if (ctx && chartData && chartData.length > 0) {{
                const chartCtx = ctx.getContext('2d');
                
                const datasets = [{{
                    label: 'ETH PRICE',
                    data: chartPrices,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0,255,0,0.1)',
                    borderWidth: 3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    fill: true,
                }}];
                
                // Add signal markers as separate datasets for better visibility
                if (buySignals.length > 0) {{
                    const buyData = new Array(chartPrices.length).fill(null);
                    buySignals.forEach(s => {{
                        buyData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: 'BUY SIGNALS',
                        data: buyData,
                        borderColor: '#0055ff',
                        backgroundColor: '#0055ff',
                        pointRadius: 10,
                        pointHoverRadius: 12,
                        pointStyle: 'rect',
                        showLine: false,
                    }});
                }}
                
                if (sellSignals.length > 0) {{
                    const sellData = new Array(chartPrices.length).fill(null);
                    sellSignals.forEach(s => {{
                        sellData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: 'SELL SIGNALS',
                        data: sellData,
                        borderColor: '#ff0000',
                        backgroundColor: '#ff0000',
                        pointRadius: 10,
                        pointHoverRadius: 12,
                        pointStyle: 'rect',
                        showLine: false,
                    }});
                }}
                
                if (profitSignals.length > 0) {{
                    const profitData = new Array(chartPrices.length).fill(null);
                    profitSignals.forEach(s => {{
                        profitData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: 'PROFIT SIGNALS',
                        data: profitData,
                        borderColor: '#ffff00',
                        backgroundColor: '#ffff00',
                        pointRadius: 10,
                        pointHoverRadius: 12,
                        pointStyle: 'rect',
                        showLine: false,
                    }});
                }}
                
                // Add zone lines if zones are defined
                if (buyZone && buyZone[0] > 0 && buyZone[1] > 0) {{
                    datasets.push({{
                        label: 'BUY ZONE (HIGH)',
                        data: chartPrices.map(() => buyZone[1]),
                        borderColor: '#00ff00',
                        borderDash: [8, 4],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                    datasets.push({{
                        label: 'BUY ZONE (LOW)',
                        data: chartPrices.map(() => buyZone[0]),
                        borderColor: '#00ff00',
                        borderDash: [8, 4],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                }}
                
                if (sellZone && sellZone[0] > 0 && sellZone[1] > 0) {{
                    datasets.push({{
                        label: 'SELL ZONE (HIGH)',
                        data: chartPrices.map(() => sellZone[1]),
                        borderColor: '#ff0000',
                        borderDash: [8, 4],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                    datasets.push({{
                        label: 'SELL ZONE (LOW)',
                        data: chartPrices.map(() => sellZone[0]),
                        borderColor: '#ff0000',
                        borderDash: [8, 4],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                }}
                
                new Chart(chartCtx, {{
                    type: 'line',
                    data: {{
                        labels: chartLabels,
                        datasets: datasets
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                labels: {{ color: '#f0f0f0' }},
                                display: true,
                                position: 'top'
                            }},
                            tooltip: {{
                                mode: 'index',
                                intersect: false,
                                callbacks: {{
                                    title: function(context) {{
                                        return context[0].label;
                                    }},
                                    label: function(context) {{
                                        const datasetLabel = context.dataset.label || '';
                                        const value = context.parsed.y;
                                        if (datasetLabel.includes('Signal')) {{
                                            // Find signal details
                                            const signalTime = new Date(chartLabels[context.dataIndex]);
                                            const signal = signalsData.find(s => {{
                                                const sTime = new Date(s.time * 1000);
                                                return Math.abs(sTime - signalTime) < 1000 * 60 * 60;
                                            }});
                                            if (signal) {{
                                                return `${{datasetLabel}}: ${{value.toFixed(2)}} | ${{signal.action}}`;
                                            }}
                                        }}
                                        return `${{datasetLabel}}: ${{value.toFixed(2)}}`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                ticks: {{ color: '#a0a0a0', maxRotation: 45, minRotation: 45 }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            y: {{
                                ticks: {{ color: '#a0a0a0' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                                title: {{
                                    display: true,
                                    text: 'Price (USDT)',
                                    color: '#a0a0a0'
                                }}
                            }}
                        }},
                        interaction: {{
                            mode: 'nearest',
                            axis: 'x',
                            intersect: false
                        }}
                    }}
                }});
            }} else {{
                // Show message if no data
                ctx.parentElement.innerHTML = '<p style="text-align: center; color: #888; padding: 40px;">Loading chart data... Please wait for bot to collect market data.</p>';
            }}
            
            async function loadSignalsFromSupabase() {{
                try {{
                    const response = await fetch('/api/signals/supabase?limit=100');
                    const data = await response.json();
                    
                    if (data.error) {{
                        alert('Error: ' + data.error);
                        return;
                    }}
                    
                    const tbody = document.getElementById('signalsTableBody');
                    if (data.signals && data.signals.length > 0) {{
                        tbody.innerHTML = data.signals.map(s => {{
                            const date = new Date(s.timestamp * 1000);
                            const actionColor = s.action.includes('BUY') ? '#22c55e' : s.action.includes('SELL') ? '#ef4444' : '#f59e0b';
                            const signalData = s.signal_data || {{}};
                            const loanAmt = signalData.loan_amount ? parseFloat(signalData.loan_amount) : null;
                            const ethAmt = signalData.eth_amount ? parseFloat(signalData.eth_amount) : null;
                            const details = loanAmt ? `Loan: $` + loanAmt.toFixed(2) : 
                                          ethAmt ? `ETH: ` + ethAmt.toFixed(4) : '';
                            const ethPrice = parseFloat(s.eth_price || 0);
                            const rsi4h = signalData.reason && signalData.reason.eth_rsi_4h ? parseFloat(signalData.reason.eth_rsi_4h).toFixed(2) : 'N/A';
                            const macdHist = signalData.reason && signalData.reason.eth_macd_hist ? parseFloat(signalData.reason.eth_macd_hist).toFixed(2) : 'N/A';
                            
                            return `
                                <tr>
                                    <td>${{date.toLocaleString()}}</td>
                                    <td><strong style="color: ${{actionColor}}">${{s.action}}</strong></td>
                                    <td>$` + ethPrice.toFixed(2) + `</td>
                                    <td>` + rsi4h + `</td>
                                    <td>` + macdHist + `</td>
                                    <td>` + details + `</td>
                                </tr>
                            `;
                        }}).join('');
                        
                        document.getElementById('signalCount').textContent = `Showing ${{data.total}} signals from database`;
                    }} else {{
                        tbody.innerHTML = '<tr><td colspan="6">No signals in database</td></tr>';
                    }}
                }} catch (error) {{
                    alert('Error loading signals: ' + error.message);
                }}
            }}
            
            // Set default dates (3 months ago to today)
            function setDefaultDates() {{
                const today = new Date();
                // Create a new Date object for yesterday
                const yesterday = new Date(today);
                yesterday.setDate(yesterday.getDate() - 1);
                const threeMonthsAgo = new Date();
                threeMonthsAgo.setMonth(today.getMonth() - 1);
                
                const formatDate = (date) => {{
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    return `${{year}}-${{month}}-${{day}}`;
                }};
                
                const startInput = document.getElementById('backtestStart');
                const endInput = document.getElementById('backtestEnd');
                
                if (!startInput.value) {{
                    startInput.value = formatDate(threeMonthsAgo);
                }}
                if (!endInput.value) {{
                    endInput.value = formatDate(yesterday);
                }}
            }}
            
            // Show/hide config update input based on mode
            function toggleConfigUpdateInput(show) {{
                const container = document.getElementById('configUpdateContainer');
                if (container) {{
                    container.style.display = show ? 'block' : 'none';
                }}
            }}
            
            // Initialize on page load
            setDefaultDates();
            
            async function runBacktest(useRealtime = false) {{
                const start = document.getElementById('backtestStart').value;
                const end = document.getElementById('backtestEnd').value;
                if (!start || !end) {{ 
                    alert('Please select start and end dates'); 
                    return; 
                }}
                
                // Validate dates
                if (new Date(start) >= new Date(end)) {{
                    alert('Start date must be before end date');
                    return;
                }}
                
                if (new Date(end) > new Date()) {{
                    alert('End date cannot be in the future');
                    return;
                }}
                
                const modeText = useRealtime ? 'Real-Time Config' : 'Historical Config';
                document.getElementById('backtestResults').innerHTML = `<p style="color: #ffd166;">Running backtest with ${{modeText}}...</p>`;
                
                try {{
                    const realtimeParam = useRealtime ? '&realtime=true' : '';
                    const updateCandles = useRealtime ? '' : `&update_candles=${{document.getElementById('configUpdateCandles')?.value || 24}}`;
                    const response = await fetch(`/api/backtest?start=${{start}}&end=${{end}}${{realtimeParam}}${{updateCandles}}`);
                    const data = await response.json();
                    
                    if (data.error) {{
                        document.getElementById('backtestResults').innerHTML = `<p style="color: #ef4444;">Error: ${{data.error}}</p>`;
                        return;
                    }}
                    
                    // Calculate cycle summary
                    let totalLoans = 0;
                    let totalEthBought = 0;
                    let totalEthSold = 0;
                    let totalRevenue = 0;
                    let totalEthGained = 0;
                    let totalProfit = 0;
                    let completedCycles = 0;
                    
                    if (data.cycles && data.cycles.length > 0) {{
                        data.cycles.forEach(cycle => {{
                            if (!cycle.incomplete) {{
                                totalLoans += cycle.total_loan || 0;
                                totalEthBought += cycle.total_eth_bought || 0;
                                totalEthSold += cycle.eth_sold || 0;
                                totalRevenue += cycle.revenue || 0;
                                totalEthGained += cycle.eth_gained || 0;
                                totalProfit += cycle.profit_usdt || 0;
                                completedCycles++;
                            }}
                        }});
                    }}
                    
                    const avgBuyPrice = totalEthBought > 0 ? totalLoans / totalEthBought : 0;
                    
                    document.getElementById('backtestResults').innerHTML = `
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px;">
                            <div><strong>Return:</strong> <span style="color: ${{data.return_pct >= 0 ? '#22c55e' : '#ef4444'}}">${{data.return_pct.toFixed(2)}}%</span></div>
                            <div><strong>Final ETH:</strong> ${{data.final_eth.toFixed(4)}}</div>
                            <div><strong>Signals:</strong> ${{data.total_signals}}</div>
                            <div><strong>Max DD:</strong> ${{data.max_drawdown.toFixed(2)}}%</div>
                            <div><strong>Cycles:</strong> ${{completedCycles}}</div>
                            <div><strong>ETH Gained:</strong> ${{totalEthGained.toFixed(4)}}</div>
                        </div>
                    `;
                    
                    // Display cycles
                    if (data.cycles && data.cycles.length > 0) {{
                        let cyclesHtml = '<div style="margin-top: 20px;"><h3 style="color: #ffd166; margin-bottom: 15px;">📊 Trading Cycles</h3>';
                        
                        data.cycles.forEach((cycle, idx) => {{
                            if (cycle.incomplete) {{
                                cyclesHtml += `
                                    <div style="background: rgba(136,136,136,0.2); padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #888;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                                            <div>
                                                <strong style="color: #888;">Cycle ${{idx + 1}} (Incomplete)</strong><br>
                                                <span style="color: #a0a0a0;">Loan: $` + parseFloat(cycle.total_loan || 0).toFixed(2) + ` | Buy: ` + parseFloat(cycle.total_eth_bought || 0).toFixed(4) + ` ETH at $` + parseFloat(cycle.avg_buy_price || 0).toFixed(2) + `</span>
                                            </div>
                                            <div style="color: #888;">→ Pending...</div>
                                        </div>
                                    </div>
                                `;
                            }} else {{
                                // Format cycle display
                                let loansDisplay = '';
                                if (cycle.loans && cycle.loans.length > 0) {{
                                    loansDisplay = cycle.loans.map((l, i) => {{
                                        const loanAmt = parseFloat(l.loan_amount || 0);
                                        const ethBought = parseFloat(l.eth_bought || 0);
                                        const buyPrice = parseFloat(l.buy_price || 0);
                                        return '|Loan $' + loanAmt.toFixed(2) + '$ Buy ' + ethBought.toFixed(4) + 'ETH at $' + buyPrice.toFixed(2) + '$|';
                                    }}).join('<br>');
                                }}
                                
                                const sellDisplay = '| Sell ' + parseFloat(cycle.eth_sold || 0).toFixed(4) + 'ETH at $' + parseFloat(cycle.sell_price || 0).toFixed(2) + '$ get $' + parseFloat(cycle.revenue || 0).toFixed(2) + '$ to Repay|';
                                const ethGainedDisplay = '| +' + parseFloat(cycle.eth_gained || 0).toFixed(4) + ' ETH to Balance                                       |';
                                const totalDisplay = '|Total $' + parseFloat(cycle.total_loan || 0).toFixed(2) + '$ Buy ' + parseFloat(cycle.total_eth_bought || 0).toFixed(4) + 'ETH at $' + parseFloat(cycle.avg_buy_price || 0).toFixed(2) + '$   |';
                                
                                cyclesHtml += `
                                    <div style="background: rgba(34,197,94,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #22c55e;">
                                        <div style="margin-bottom: 10px;">
                                            <strong style="color: #22c55e;">Cycle ${{idx + 1}}</strong>
                                        </div>
                                        <div style="font-family: monospace; font-size: 0.9em; line-height: 1.8;">
                                            <div style="color: #a0a0a0; margin-bottom: 5px;">${{loansDisplay}}</div>
                                            <div style="color: #ef4444; margin-bottom: 5px;">→ ${{sellDisplay}}</div>
                                            <div style="color: #ffd166; margin-bottom: 5px;">→ ${{ethGainedDisplay}}</div>
                                            <div style="color: #22c55e; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">${{totalDisplay}}</div>
                                        </div>
                                        <div style="margin-top: 10px; text-align: right;">
                                            <span style="color: ${{cycle.profit_usdt >= 0 ? '#22c55e' : '#ef4444'}};"><strong>Profit:</strong> $` + parseFloat(cycle.profit_usdt || 0).toFixed(2) + `</span>
                                        </div>
                                    </div>
                                `;
                            }}
                        }});
                        
                        // Add summary
                        if (completedCycles > 0) {{
                            cyclesHtml += `
                                <div style="background: rgba(255,209,102,0.2); padding: 15px; border-radius: 8px; margin-top: 20px; border: 2px solid #ffd166;">
                                    <h4 style="color: #ffd166; margin-bottom: 10px;">📈 Total Summary</h4>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                        <div><strong>Total Loans:</strong> $` + parseFloat(totalLoans).toFixed(2) + `</div>
                                        <div><strong>Total ETH Bought:</strong> ` + parseFloat(totalEthBought).toFixed(4) + ` ETH</div>
                                        <div><strong>Avg Buy Price:</strong> $` + parseFloat(avgBuyPrice).toFixed(2) + `</div>
                                        <div><strong>Total ETH Sold:</strong> ` + parseFloat(totalEthSold).toFixed(4) + ` ETH</div>
                                        <div><strong>Total Revenue:</strong> $` + parseFloat(totalRevenue).toFixed(2) + `</div>
                                        <div><strong>Total ETH Gained:</strong> ` + parseFloat(totalEthGained).toFixed(4) + ` ETH</div>
                                        <div><strong>Total Profit:</strong> <span style="color: ${{totalProfit >= 0 ? '#22c55e' : '#ef4444'}}">$` + parseFloat(totalProfit).toFixed(2) + `</span></div>
                                        <div><strong>Completed Cycles:</strong> ${{completedCycles}}</div>
                                    </div>
                                </div>
                            `;
                        }}
                        
                        cyclesHtml += '</div>';
                        document.getElementById('backtestResults').innerHTML += cyclesHtml;
                    }}
                    
                    if (data.equity_curve && data.equity_curve.length > 0) {{
                        document.getElementById('backtestChartContainer').style.display = 'block';
                        const ctx = document.getElementById('backtestChart').getContext('2d');
                        
                        // Prepare signal markers - arrays for equity values at signal points
                        const buySignals = new Array(data.equity_curve.length).fill(null);
                        const sellSignals = new Array(data.equity_curve.length).fill(null);
                        const profitSignals = new Array(data.equity_curve.length).fill(null);
                        
                        // Map signals to chart indices using signal_indices
                        if (data.signal_indices && Object.keys(data.signal_indices).length > 0) {{
                            Object.keys(data.signal_indices).forEach(idxStr => {{
                                const idx = parseInt(idxStr);
                                if (idx >= 0 && idx < data.equity_curve.length) {{
                                    const signalsAtIdx = data.signal_indices[idxStr];
                                    signalsAtIdx.forEach(signal => {{
                                        const equityValue = data.equity_curve[idx];
                                        if (signal.action === 'BUY_ETH_WITH_LOAN') {{
                                            buySignals[idx] = equityValue;
                                        }} else if (signal.action === 'SELL_ETH_REPAY_LOAN') {{
                                            sellSignals[idx] = equityValue;
                                        }} else if (signal.action === 'TAKE_PROFIT_SELL') {{
                                            profitSignals[idx] = equityValue;
                                        }}
                                    }});
                                }}
                            }});
                        }}
                        
                        const datasets = [{{
                            label: 'EQUITY CURVE',
                            data: data.equity_curve,
                            borderColor: '#00ff00',
                            backgroundColor: 'rgba(0,255,0,0.1)',
                            borderWidth: 3,
                            fill: true,
                            pointRadius: 2,
                        }}];
                        
                        // Add price line if available
                        if (data.price_history && data.price_history.length > 0) {{
                            const prices = data.price_history.map(p => p.price);
                            // Normalize prices to equity scale for visibility
                            const priceMin = Math.min(...prices);
                            const priceMax = Math.max(...prices);
                            const equityMin = Math.min(...data.equity_curve);
                            const equityMax = Math.max(...data.equity_curve);
                            const priceScale = (equityMax - equityMin) / (priceMax - priceMin) * 0.3; // 30% of chart height
                            const normalizedPrices = prices.map(p => equityMin + (p - priceMin) * priceScale);
                            
                            datasets.push({{
                                label: 'ETH PRICE (SCALED)',
                                data: normalizedPrices,
                                borderColor: '#00ff88',
                                borderWidth: 2,
                                borderDash: [8, 4],
                                pointRadius: 0,
                                yAxisID: 'y',
                            }});
                        }}
                        
                        // Add signal markers
                        if (buySignals.some(v => v !== null)) {{
                            datasets.push({{
                                label: 'BUY SIGNALS',
                                data: buySignals,
                                borderColor: '#0055ff',
                                backgroundColor: '#0055ff',
                                pointRadius: 8,
                                pointHoverRadius: 10,
                                pointStyle: 'rect',
                                showLine: false,
                            }});
                        }}
                        
                        if (sellSignals.some(v => v !== null)) {{
                            datasets.push({{
                                label: 'SELL SIGNALS',
                                data: sellSignals,
                                borderColor: '#ff0000',
                                backgroundColor: '#ff0000',
                                pointRadius: 8,
                                pointHoverRadius: 10,
                                pointStyle: 'rect',
                                showLine: false,
                            }});
                        }}
                        
                        if (profitSignals.some(v => v !== null)) {{
                            datasets.push({{
                                label: 'PROFIT SIGNALS',
                                data: profitSignals,
                                borderColor: '#ffff00',
                                backgroundColor: '#ffff00',
                                pointRadius: 8,
                                pointHoverRadius: 10,
                                pointStyle: 'rect',
                                showLine: false,
                            }});
                        }}
                        
                        // Destroy existing chart if any
                        if (window.backtestChartInstance) {{
                            window.backtestChartInstance.destroy();
                        }}
                        
                        window.backtestChartInstance = new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: data.equity_curve.map((_, i) => {{
                                    if (data.price_history && data.price_history[i]) {{
                                        return new Date(data.price_history[i].timestamp * 1000).toLocaleDateString();
                                    }}
                                    return `Candle ${{i+1}}`;
                                }}),
                                datasets: datasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{
                                        labels: {{ color: '#f0f0f0' }},
                                        display: true,
                                        position: 'top'
                                    }},
                                    tooltip: {{
                                        mode: 'index',
                                        intersect: false,
                                        callbacks: {{
                                            title: function(context) {{
                                                return context[0].label;
                                            }},
                                            label: function(context) {{
                                                const label = context.dataset.label || '';
                                                const value = context.parsed.y;
                                                const idx = context.dataIndex;
                                                
                                                if (label.includes('Signal')) {{
                                                    // Find signal at this index
                                                    const signalIdx = data.signal_indices ? Object.keys(data.signal_indices).find(k => parseInt(k) === idx) : null;
                                                    if (signalIdx && data.signal_indices[signalIdx] && data.signal_indices[signalIdx].length > 0) {{
                                                        const signal = data.signal_indices[signalIdx][0];
                                                        const price = signal.eth_price || 0;
                                                        const signalInfo = signal.signal || {{}};
                                                        let details = `${{label}}: Equity ${{value.toFixed(2)}} | Price $` + parseFloat(price).toFixed(2);
                                                        
                                                        if (signalInfo.loan_amount) {{
                                                            details += ` | Loan: $` + parseFloat(signalInfo.loan_amount).toFixed(2);
                                                        }}
                                                        if (signalInfo.eth_amount) {{
                                                            details += ` | ETH: ` + parseFloat(signalInfo.eth_amount).toFixed(4);
                                                        }}
                                                        return details;
                                                    }}
                                                }}
                                                
                                                if (label.includes('Price')) {{
                                                    return `${{label}}: ${{value.toFixed(2)}} (scaled)`;
                                                }}
                                                
                                                return `${{label}}: ${{value.toFixed(2)}}`;
                                            }}
                                        }}
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        ticks: {{ color: '#a0a0a0', maxRotation: 45, minRotation: 45 }},
                                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                                    }},
                                    y: {{
                                        ticks: {{ color: '#a0a0a0' }},
                                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                                        title: {{
                                            display: true,
                                            text: 'Equity (USDT)',
                                            color: '#a0a0a0'
                                        }}
                                    }}
                                }},
                                interaction: {{
                                    mode: 'nearest',
                                    axis: 'x',
                                    intersect: false
                                }}
                            }}
                        }});
                    }}
                }} catch (error) {{
                    document.getElementById('backtestResults').innerHTML = `<p style="color: #ef4444;">Error: ${{error.message}}</p>`;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.get("/api/backtest", response_class=JSONResponse)
def api_backtest(start: str, end: str, realtime: bool = False, update_candles: int = 24):
    try:
        # Use Binance client for backtesting (better historical data)
        backtester = RotationBacktester(BinanceClient())
        results = backtester.run_backtest(start, end, use_realtime_config=realtime, config_update_candles=update_candles)
        if "error" in results:
            return JSONResponse(content={"error": results["error"]}, status_code=400)
        return sanitize_for_json(results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/state", response_class=JSONResponse)
def api_state():
    return sanitize_for_json({
        "market": LAST_MARKET_STATE,
        "config": LAST_CONFIG,
        "position": asdict(bot.position),
        "recent_signals": RECENT_SIGNALS[-50:],
    })

@app.get("/api/signals", response_class=JSONResponse)
def api_signals(limit: int = 100):
    """Get signals from memory (recent)"""
    return sanitize_for_json({
        "signals": RECENT_SIGNALS[-limit:],
        "total": len(RECENT_SIGNALS),
    })

@app.get("/api/signals/supabase", response_class=JSONResponse)
def api_signals_supabase(limit: int = 100):
    """Get signals from Supabase database"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return JSONResponse(content={"error": "Supabase not configured"}, status_code=400)
    
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "order": "timestamp.desc",
        "limit": limit,
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return sanitize_for_json({
            "signals": data,
            "total": len(data),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

