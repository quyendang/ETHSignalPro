import time
import os
import hmac
import hashlib
import requests
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from math import isnan

#####################################
# CONFIG ENV FOR RENDER
#####################################

BINANCE_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_DEVICE = os.getenv("PUSHOVER_DEVICE", "")

#####################################
# Pushover notify
#####################################

def _pushover_notify(title: str, message: str):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        print("[WARN] Pushover credentials missing.")
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
    except Exception as e:
        print("[PUSHOVER ERROR]", e)

#####################################
# 1. TA INDICATORS
#####################################

def calc_rsi(closes: List[float], period=14):
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


def calc_ema(series: np.ndarray, period: int):
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
    tr_list = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        tr_list.append(tr)
    tr_arr = np.array(tr_list)
    return float(tr_arr[-period:].mean())


def realized_vol(closes: List[float], lookback=60):
    if len(closes) < lookback + 2:
        return float("nan")
    arr = np.array(closes[-lookback:], dtype=float)
    rets = np.diff(np.log(arr))
    return float(np.std(rets))


def qtile(values: List[float], q: float):
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(np.array(values, dtype=float), q))


#####################################
# 2. BINANCE CLIENT
#####################################

class BinanceClient:
    BASE_URL = "https://api.binance.com"

    def _sign(self, query: str):
        return hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _get(self, path, params=None, signed=False):
        url = self.BASE_URL + path
        headers = {"X-MBX-APIKEY": BINANCE_KEY}
        if params is None:
            params = {}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query = "&".join(f"{k}={v}" for k, v in params.items())
            params["signature"] = self._sign(query)
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("[BINANCE ERROR]", e)
            return None

    ##############################
    # Public endpoints
    ##############################

    def get_ticker(self, symbol: str):
        r = self._get("/api/v3/ticker/price", {"symbol": symbol})
        return {"last": float(r["price"])} if r else {"last": float("nan")}

    def get_ohlcv(self, symbol: str, interval: str, limit=500):
        r = self._get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if not r:
            return []
        candles = []
        for c in r:
            candles.append({
                "timestamp": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return candles

    ##############################
    # Private endpoints
    ##############################

    def get_balance(self, asset: str):
        r = self._get("/api/v3/account", signed=True)
        if not r:
            return 0.0
        for bal in r.get("balances", []):
            if bal["asset"] == asset:
                return float(bal["free"])
        return 0.0


#####################################
# 3. DATA STRUCT
#####################################

@dataclass
class MarketState:
    eth_price: float
    btc_price: float
    ethbtc_price: float

    eth_rsi_4h: float
    btc_rsi_4h: float

    eth_macd_hist_4h: float
    btc_macd_hist_4h: float


@dataclass
class PositionState:
    spot_eth: float
    loan_usdt: float
    avg_entry_eth: float
    last_sell_price: Optional[float] = None
    last_sell_size: Optional[float] = None


#####################################
# 4. HISTORY PROVIDER
#####################################

class HistoryProvider:
    def __init__(self, client: BinanceClient):
        self.client = client

    def closes(self, symbol, interval, limit):
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["close"] for c in candles]

    def highs(self, symbol, interval, limit):
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["high"] for c in candles]

    def lows(self, symbol, interval, limit):
        candles = self.client.get_ohlcv(symbol, interval, limit)
        return [c["low"] for c in candles]

    def rsi_series(self, symbol, interval, lookback):
        closes = self.closes(symbol, interval, lookback+30)
        out = []
        for i in range(20, len(closes)):
            out.append(calc_rsi(closes[:i]))
        return out


#####################################
# 5. DYNAMIC CONFIG
#####################################

class DynamicConfigBuilder:
    def __init__(self, hist: HistoryProvider):
        self.h = hist

    def build(self):
        # RSI dynamic
        eth_rsi_4h = self.h.rsi_series("ETHUSDT", "4h", 180)
        btc_rsi_4h = self.h.rsi_series("BTCUSDT", "4h", 180)
        eth_overbought = qtile(eth_rsi_4h, 0.8)
        eth_oversold = qtile(eth_rsi_4h, 0.2)
        btc_overbought = qtile(btc_rsi_4h, 0.8)
        btc_oversold = qtile(btc_rsi_4h, 0.2)

        # BTC dump level
        daily = self.h.client.get_ohlcv("BTCUSDT", "1d", 60)
        if not daily:
            btc_dump = 90000
        else:
            closes = [c["close"] for c in daily]
            highs = [c["high"] for c in daily]
            lows = [c["low"] for c in daily]
            if len(closes) < 20:
                btc_dump = closes[-1]*0.92
            else:
                ma20 = sum(closes[-20:]) / 20
                atr14 = calc_atr(highs, lows, closes)
                btc_dump = ma20 - 1.5 * atr14

        # Buyback zone ETH
        eth_4h = self.h.client.get_ohlcv("ETHUSDT", "4h", 120)
        closes_4h = [c["close"] for c in eth_4h]
        recent = closes_4h[-60:]
        lo = min(recent)
        hi = max(recent)
        fib50 = lo + 0.5*(hi-lo)
        fib618 = lo + 0.618*(hi-lo)
        buy_low = (lo+fib50)/2
        buy_high = fib618

        # Vol → max LTV
        vol = realized_vol(closes_4h)
        if isnan(vol):
            max_ltv = 0.3
        elif vol < 0.02:
            max_ltv = 0.4
        elif vol < 0.04:
            max_ltv = 0.3
        else:
            max_ltv = 0.2

        # ETHBTC breakdown
        ethbtc = self.h.client.get_ohlcv("ETHBTC", "4h", 200)
        ethbtc_closes = [c["close"] for c in ethbtc]
        ethbtc_highs = [c["high"] for c in ethbtc]
        ethbtc_lows = [c["low"] for c in ethbtc]

        if len(ethbtc_closes) >= 60:
            ma50 = sum(ethbtc_closes[-50:]) / 50
            atr_ethbtc = calc_atr(ethbtc_highs, ethbtc_lows, ethbtc_closes)
            ethbtc_break = ma50 - 0.5 * atr_ethbtc
        else:
            ethbtc_break = ethbtc_closes[-1]*0.97

        # Sell min price
        sell_min = hi - 0.2*(hi-lo)

        return {
            "eth_rsi_overbought_4h": eth_overbought,
            "eth_rsi_oversold_4h": eth_oversold,
            "btc_rsi_overbought_4h": btc_overbought,
            "btc_rsi_oversold_4h": btc_oversold,
            "btc_dump_level": btc_dump,
            "buyback_zone": (buy_low, buy_high),
            "max_ltv": max_ltv,
            "ethbtc_breakdown": ethbtc_break,
            "sell_spot_eth_price_min": sell_min,
        }


#####################################
# 6. STRATEGY
#####################################

class EthStrategy:
    def __init__(self, conf):
        self.conf = conf
        self.prev_eth_macd_hist = None

    def update_conf(self, conf):
        self.conf = conf

    def should_sell_spot(self, m: MarketState, p: PositionState):
        if m.eth_price < self.conf["sell_spot_eth_price_min"]:
            self.prev_eth_macd_hist = m.eth_macd_hist_4h
            return None

        rsi_hot = m.eth_rsi_4h >= self.conf["eth_rsi_overbought_4h"]
        macd_peak = False
        if self.prev_eth_macd_hist is not None:
            macd_peak = (
                self.prev_eth_macd_hist > 0
                and m.eth_macd_hist_4h > 0
                and m.eth_macd_hist_4h < self.prev_eth_macd_hist
            )

        bad_ethbtc = m.ethbtc_price < self.conf["ethbtc_breakdown"]
        bad_btc = m.btc_price < self.conf["btc_dump_level"]

        self.prev_eth_macd_hist = m.eth_macd_hist_4h

        if (rsi_hot or macd_peak) and (bad_ethbtc or bad_btc):
            size = min(30, p.spot_eth*0.3)
            if size <= 0:
                return None
            return {
                "action": "SELL_SPOT_ETH",
                "size": size,
                "reason": {
                    "rsi": m.eth_rsi_4h,
                    "macd_hist": m.eth_macd_hist_4h,
                    "ethbtc": m.ethbtc_price,
                    "btc": m.btc_price
                },
            }
        return None

    def should_buyback_spot(self, m: MarketState, p: PositionState):
        if not p.last_sell_size:
            return None

        low, high = self.conf["buyback_zone"]
        if not(low <= m.eth_price <= high):
            return None

        if m.btc_price < self.conf["btc_dump_level"]*0.95:
            return None

        if m.eth_rsi_4h > self.conf["eth_rsi_overbought_4h"]:
            return None

        return {
            "action": "BUYBACK_SPOT_ETH",
            "size": p.last_sell_size,
            "reason": {
                "eth": m.eth_price,
                "zone": (low, high),
                "btc": m.btc_price,
            },
        }

    def max_safe_loan(self, m: MarketState, p: PositionState):
        if p.spot_eth <= 0:
            return 0
        coll_value = p.spot_eth * m.eth_price
        max_ltv = self.conf["max_ltv"]
        current_ltv = p.loan_usdt / coll_value if p.loan_usdt > 0 else 0
        room = max_ltv - current_ltv
        if room <= 0:
            return 0
        return coll_value * room

    def should_open_loan(self, m, p):
        low, high = self.conf["buyback_zone"]
        if m.eth_price > high:
            return None
        if m.btc_price < self.conf["btc_dump_level"]:
            return None
        if m.ethbtc_price < self.conf["ethbtc_breakdown"]:
            return None
        loan = self.max_safe_loan(m, p)
        if loan <= 0:
            return None
        loan = min(loan, 50_000)
        return {
            "action": "OPEN_LOAN_BUY_ETH",
            "loan_amount": loan,
            "reason": {
                "eth": m.eth_price,
                "btc": m.btc_price,
                "ethbtc": m.ethbtc_price,
            },
        }

    def should_repay_loan(self, m, p):
        if p.loan_usdt <= 0:
            return None
        low, high = self.conf["buyback_zone"]
        target = high*1.10
        if m.eth_price < target:
            return None
        if m.eth_rsi_4h < self.conf["eth_rsi_overbought_4h"]:
            return None
        size = (p.loan_usdt*1.01)/m.eth_price
        return {
            "action": "REPAY_LOAN_SELL_ETH",
            "size": size,
            "reason": {
                "eth": m.eth_price,
                "loan": p.loan_usdt,
                "target": target,
            },
        }


#####################################
# 7. ETH BOT MAIN
#####################################

class EthBot:
    def __init__(self):
        self.client = BinanceClient()
        self.hist = HistoryProvider(self.client)
        self.conf_builder = DynamicConfigBuilder(self.hist)
        self.conf = self.conf_builder.build()
        self.strategy = EthStrategy(self.conf)

        self.position = PositionState(
            spot_eth=0.0,
            loan_usdt=0.0,
            avg_entry_eth=3150.0,
        )

        self.last_conf_update = 0
        self.update_interval = 4*60*60  # update dynamic config mỗi 4h

    def update_position(self):
        self.position.spot_eth = self.client.get_balance("ETH")

    def build_market_state(self):
        eth = self.client.get_ticker("ETHUSDT")["last"]
        btc = self.client.get_ticker("BTCUSDT")["last"]
        ethbtc = self.client.get_ticker("ETHBTC")["last"]

        eth4 = self.hist.closes("ETHUSDT", "4h", 70)
        btc4 = self.hist.closes("BTCUSDT", "4h", 70)

        eth_rsi = calc_rsi(eth4)
        btc_rsi = calc_rsi(btc4)

        _, _, eth_hist = calc_macd(eth4)
        _, _, btc_hist = calc_macd(btc4)

        return MarketState(
            eth_price=eth,
            btc_price=btc,
            ethbtc_price=ethbtc,
            eth_rsi_4h=eth_rsi,
            btc_rsi_4h=btc_rsi,
            eth_macd_hist_4h=eth_hist,
            btc_macd_hist_4h=btc_hist,
        )

    def maybe_update_config(self):
        if time.time() - self.last_conf_update > self.update_interval:
            self.conf = self.conf_builder.build()
            self.strategy.update_conf(self.conf)
            self.last_conf_update = time.time()
            _pushover_notify("ETH Bot – Config Updated", str(self.conf))

    def handle_signal(self, signal: Dict[str, Any]):
        action = signal["action"]
        txt = f"{action}\n{signal}"
        print("[SIGNAL]", txt)
        _pushover_notify("ETH SIGNAL", txt)

        # update position only internal
        if action == "SELL_SPOT_ETH":
            self.position.last_sell_price = signal["reason"]
            self.position.last_sell_size = signal["size"]
        elif action == "BUYBACK_SPOT_ETH":
            self.position.last_sell_price = None
            self.position.last_sell_size = None

        elif action == "OPEN_LOAN_BUY_ETH":
            self.position.loan_usdt += signal["loan_amount"]

        elif action == "REPAY_LOAN_SELL_ETH":
            self.position.loan_usdt = 0

    def run_step(self):
        self.maybe_update_config()
        self.update_position()
        m = self.build_market_state()

        # order priority
        signal = self.strategy.should_repay_loan(m, self.position)
        if not signal:
            signal = self.strategy.should_sell_spot(m, self.position)
        if not signal:
            signal = self.strategy.should_buyback_spot(m, self.position)
        if not signal:
            signal = self.strategy.should_open_loan(m, self.position)

        if signal:
            self.handle_signal(signal)

    def run_forever(self):
        while True:
            try:
                self.run_step()
            except Exception as e:
                print("[ERR]", e)
                _pushover_notify("ETH BOT ERROR", str(e))
            time.sleep(60)  # check mỗi phút


#####################################
# START BOT
#####################################

if __name__ == "__main__":
    bot = EthBot()
    bot.run_forever()
