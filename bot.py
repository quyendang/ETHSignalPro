import os
import time
import threading
import json
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
# Bybit Client
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
    def __init__(self, client: BybitClient):
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

    def should_buy_with_loan(self, m: MarketState, p: PositionState) -> Optional[Dict]:
        """Buy ETH using margin loan when conditions are favorable"""
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
        
        # Calculate loan amount (conservative: 50% of available)
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

    def should_sell_to_repay(self, m: MarketState, p: PositionState) -> Optional[Dict]:
        """Sell ETH to repay loan when profitable"""
        if p.loan_usdt <= 0:
            return None
        
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
        self.client = BybitClient()
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
        
        # Check signals in priority order
        signal = self.strategy.should_sell_to_repay(m, self.position)
        if not signal:
            signal = self.strategy.should_take_profit(m, self.position)
        if not signal:
            signal = self.strategy.should_buy_with_loan(m, self.position)
        
        if signal:
            self.handle_signal(signal, m)

#############################
# Backtesting Module
#############################

class RotationBacktester:
    def __init__(self, client: BybitClient):
        self.client = client

    def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
        
        all_candles = []
        current_start = start_ts * 1000
        end_ms = end_ts * 1000
        
        interval_ms = 4 * 60 * 60 * 1000 if interval == "4h" else 24 * 60 * 60 * 1000
        
        while current_start < end_ms:
            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": self.client._map_interval(interval),
                "start": current_start,
                "end": end_ms,
                "limit": 200,
            }
            data = self.client._get("/v5/market/kline", params)
            if not data or "result" not in data or not data["result"].get("list"):
                break
            
            batch = []
            for item in data["result"]["list"]:
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
                break
            
            all_candles.extend(batch)
            current_start = max(c["timestamp"] * 1000 for c in batch) + interval_ms
            
            if len(batch) < 200:
                break
        
        all_candles.sort(key=lambda x: x["timestamp"])
        return all_candles

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        print(f"[BACKTEST] Starting from {start_date} to {end_date}")
        
        eth_4h = self.get_historical_data("ETHUSDT", "4h", start_date, end_date)
        eth_1d = self.get_historical_data("ETHUSDT", "1d", start_date, end_date)
        btc_4h = self.get_historical_data("BTCUSDT", "4h", start_date, end_date)
        btc_1d = self.get_historical_data("BTCUSDT", "1d", start_date, end_date)
        
        if len(eth_4h) < 100:
            return {"error": "Not enough data"}
        
        position = PositionState(
            spot_eth=INITIAL_ETH,
            loan_usdt=0.0,
            avg_entry_price=INITIAL_ENTRY_PRICE,
        )
        
        conf_builder = DynamicConfigBuilder(self.client)
        conf = conf_builder.build()
        strategy = EthRotationStrategy(conf)
        
        signals = []
        equity_curve = []
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
            if i % 24 == 0:
                try:
                    conf = conf_builder.build()
                    strategy.update_conf(conf)
                except:
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
            
            # Check signals
            signal = strategy.should_sell_to_repay(market_state, position)
            if not signal:
                signal = strategy.should_take_profit(market_state, position)
            if not signal:
                signal = strategy.should_buy_with_loan(market_state, position)
            
            if signal:
                action = signal["action"]
                
                if action == "BUY_ETH_WITH_LOAN":
                    loan = signal["loan_amount"]
                    position.loan_usdt += loan
                    bought_eth = loan / market_state.eth_price
                    position.spot_eth += bought_eth
                    total_value = (position.spot_eth - bought_eth) * position.avg_entry_price + loan
                    position.avg_entry_price = total_value / position.spot_eth if position.spot_eth > 0 else position.avg_entry_price
                    
                elif action == "SELL_ETH_REPAY_LOAN":
                    eth_sold = signal["eth_amount"]
                    position.spot_eth = max(0.0, position.spot_eth - eth_sold)
                    position.loan_usdt = 0.0
                    
                elif action == "TAKE_PROFIT_SELL":
                    eth_sold = signal["eth_amount"]
                    position.spot_eth = max(0.0, position.spot_eth - eth_sold)
                
                signals.append({
                    "timestamp": candle_ts,
                    "action": action,
                    "eth_price": market_state.eth_price,
                    "signal": signal,
                })
            
            current_value = position.spot_eth * market_state.eth_price - position.loan_usdt
            equity_curve.append(current_value)
            
            if current_value > max_equity:
                max_equity = current_value
            
            drawdown = ((max_equity - current_value) / max_equity) * 100 if max_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        final_value = equity_curve[-1] if equity_curve else initial_value
        return_pct = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        signal_counts = {}
        for s in signals:
            action = s["action"]
            signal_counts[action] = signal_counts.get(action, 0) + 1
        
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
            "equity_curve": equity_curve[::max(1, len(equity_curve)//1000)],  # Sample
            "signals": signals[-20:],
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
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0b1020 0%, #1a1f3a 100%);
                color: #f0f0f0;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
                border-radius: 15px;
            }}
            .header h1 {{ color: #ffd166; font-size: 2.5em; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .card {{
                background: linear-gradient(135deg, #141a33 0%, #1e2540 100%);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }}
            .card h2 {{ color: #ffd166; margin-bottom: 15px; border-bottom: 2px solid rgba(255,209,102,0.3); padding-bottom: 10px; }}
            .metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
            .metric-value {{ font-weight: bold; font-size: 1.1em; }}
            .positive {{ color: #22c55e; }}
            .negative {{ color: #ef4444; }}
            .chart-container {{ position: relative; height: 400px; margin-top: 20px; }}
            .zone-buy {{ background: rgba(34,197,94,0.2); border-left: 4px solid #22c55e; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .zone-sell {{ background: rgba(239,68,68,0.2); border-left: 4px solid #ef4444; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
            th {{ background: rgba(255,209,102,0.2); color: #ffd166; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔄 ETH Rotation Bot</h1>
                <p style="color: #a0a0a0;">Increase ETH holdings through margin rotation</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>💰 Position</h2>
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
                    <h2>📊 Market</h2>
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
                    <h2>⚙️ Dynamic Zones</h2>
                    <div class="zone-buy">
                        <strong>🟢 Buy Zone:</strong><br>
                        ${fmt(buy_zone[0], 2)} - ${fmt(buy_zone[1], 2)}
                    </div>
                    <div class="zone-sell">
                        <strong>🔴 Sell Zone:</strong><br>
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
                <h2>📈 Price Chart with Zones</h2>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>📡 Recent Signals</h2>
                <div style="margin-bottom: 15px;">
                    <button onclick="loadSignalsFromSupabase()" style="padding: 8px 16px; background: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                        Load from Database
                    </button>
                    <span id="signalCount" style="color: #a0a0a0;">Showing {len(RECENT_SIGNALS)} signals</span>
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
                            <td><strong style="color: {'#22c55e' if 'BUY' in s['action'] else '#ef4444' if 'SELL' in s['action'] else '#f59e0b'}">{s['action']}</strong></td>
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
                <h2>🧪 Backtest</h2>
                <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                    <div>
                        <label style="display: block; margin-bottom: 5px;">Start:</label>
                        <input type="date" id="backtestStart" style="padding: 8px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #f0f0f0;">
                    </div>
                    <div>
                        <label style="display: block; margin-bottom: 5px;">End:</label>
                        <input type="date" id="backtestEnd" style="padding: 8px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #f0f0f0;">
                    </div>
                    <div style="display: flex; align-items: flex-end;">
                        <button onclick="runBacktest()" style="padding: 10px 20px; background: #ffd166; color: #0b1020; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Run Backtest</button>
                    </div>
                </div>
                <div id="backtestResults"></div>
                <div class="chart-container" id="backtestChartContainer" style="display: none;">
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
                    label: 'ETH Price',
                    data: chartPrices,
                    borderColor: '#ffd166',
                    backgroundColor: 'rgba(255,209,102,0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    fill: true,
                }}];
                
                // Add signal markers as separate datasets for better visibility
                if (buySignals.length > 0) {{
                    const buyData = new Array(chartPrices.length).fill(null);
                    buySignals.forEach(s => {{
                        buyData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: '🟢 Buy Signals',
                        data: buyData,
                        borderColor: '#22c55e',
                        backgroundColor: '#22c55e',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        pointStyle: 'circle',
                        showLine: false,
                    }});
                }}
                
                if (sellSignals.length > 0) {{
                    const sellData = new Array(chartPrices.length).fill(null);
                    sellSignals.forEach(s => {{
                        sellData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: '🔴 Sell Signals',
                        data: sellData,
                        borderColor: '#ef4444',
                        backgroundColor: '#ef4444',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        pointStyle: 'circle',
                        showLine: false,
                    }});
                }}
                
                if (profitSignals.length > 0) {{
                    const profitData = new Array(chartPrices.length).fill(null);
                    profitSignals.forEach(s => {{
                        profitData[s.x] = s.y;
                    }});
                    datasets.push({{
                        label: '🟡 Profit Signals',
                        data: profitData,
                        borderColor: '#f59e0b',
                        backgroundColor: '#f59e0b',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        pointStyle: 'circle',
                        showLine: false,
                    }});
                }}
                
                // Add zone lines if zones are defined
                if (buyZone && buyZone[0] > 0 && buyZone[1] > 0) {{
                    datasets.push({{
                        label: 'Buy Zone (High)',
                        data: chartPrices.map(() => buyZone[1]),
                        borderColor: '#22c55e',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                    datasets.push({{
                        label: 'Buy Zone (Low)',
                        data: chartPrices.map(() => buyZone[0]),
                        borderColor: '#22c55e',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                }}
                
                if (sellZone && sellZone[0] > 0 && sellZone[1] > 0) {{
                    datasets.push({{
                        label: 'Sell Zone (High)',
                        data: chartPrices.map(() => sellZone[1]),
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }});
                    datasets.push({{
                        label: 'Sell Zone (Low)',
                        data: chartPrices.map(() => sellZone[0]),
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
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
            
            async function runBacktest() {{
                const start = document.getElementById('backtestStart').value;
                const end = document.getElementById('backtestEnd').value;
                if (!start || !end) {{ alert('Please select dates'); return; }}
                
                document.getElementById('backtestResults').innerHTML = '<p style="color: #ffd166;">Running...</p>';
                
                try {{
                    const response = await fetch(`/api/backtest?start=${{start}}&end=${{end}}`);
                    const data = await response.json();
                    
                    if (data.error) {{
                        document.getElementById('backtestResults').innerHTML = `<p style="color: #ef4444;">Error: ${{data.error}}</p>`;
                        return;
                    }}
                    
                    document.getElementById('backtestResults').innerHTML = `
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                            <div><strong>Return:</strong> <span style="color: ${{data.return_pct >= 0 ? '#22c55e' : '#ef4444'}}">${{data.return_pct.toFixed(2)}}%</span></div>
                            <div><strong>Final ETH:</strong> ${{data.final_eth.toFixed(4)}}</div>
                            <div><strong>Signals:</strong> ${{data.total_signals}}</div>
                            <div><strong>Max DD:</strong> ${{data.max_drawdown.toFixed(2)}}%</div>
                        </div>
                    `;
                    
                    if (data.equity_curve && data.equity_curve.length > 0) {{
                        document.getElementById('backtestChartContainer').style.display = 'block';
                        const ctx = document.getElementById('backtestChart').getContext('2d');
                        new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: data.equity_curve.map((_, i) => `Candle ${{i+1}}`),
                                datasets: [{{
                                    label: 'Equity',
                                    data: data.equity_curve,
                                    borderColor: '#ffd166',
                                    borderWidth: 2,
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{ legend: {{ labels: {{ color: '#f0f0f0' }} }} }},
                                scales: {{
                                    x: {{ ticks: {{ color: '#a0a0a0' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }},
                                    y: {{ ticks: {{ color: '#a0a0a0' }}, grid: {{ color: 'rgba(255,255,255,0.1)' }} }}
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
def api_backtest(start: str, end: str):
    try:
        backtester = RotationBacktester(bot.client)
        results = backtester.run_backtest(start, end)
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

