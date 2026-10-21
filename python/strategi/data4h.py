"""Ambil candle 4 jam (atau interval lain) (Binance, fallback Bitget futures) sejak 2021-06, disimpan ke cache lokal."""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

START = pd.Timestamp("2021-06-01", tz="UTC")
CACHE = os.path.join(os.path.dirname(__file__), "..", "hasil", "cache4h")
WATCHLIST = ["HYPE", "ASTER", "XPL", "SOL", "BTC", "TAO", "XRP", "ETH", "DOGE", "bitget:LIT", "BNB", "XAU"]


def _frame(rows, cols, step=pd.Timedelta(hours=4)):
    df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("time").astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[(df.index >= START) & (df.index + step <= pd.Timestamp.now(tz="UTC"))]


def binance(sym: str, interval: str = "4h") -> pd.DataFrame:
    rows, start = [], int(START.value // 1_000_000)
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": sym, "interval": interval, "startTime": start, "limit": 1000}, timeout=20)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        start = page[-1][0] + 1
    if not rows:
        raise ValueError("kosong")
    return _frame(rows, None, pd.Timedelta(interval))


def bitget(sym: str, interval: str = "4h") -> pd.DataFrame:
    base = "https://api.bitget.com/api/v2/mix/market/"
    p = {"symbol": sym, "productType": "USDT-FUTURES", "granularity": interval.upper()}
    rows = requests.get(base + "candles", params={**p, "limit": 200}, timeout=20).json()["data"]
    while True:
        oldest = min(int(r[0]) for r in rows)
        if oldest <= START.value // 1_000_000:
            break
        body = requests.get(base + "history-candles", params={**p, "endTime": oldest, "limit": 200}, timeout=20).json()
        older = [r for r in (body.get("data") or []) if int(r[0]) < oldest]
        if not older:
            break
        rows = older + rows
        time.sleep(0.12)
    return _frame(rows, None, pd.Timedelta(interval))


def load(symbol: str, interval: str = "4h") -> tuple[str, pd.DataFrame]:
    forced, _, rest = symbol.rpartition(":")
    sym = rest.upper() + "USDT"
    path = os.path.join(CACHE, sym + ("" if interval == "4h" else "_" + interval) + ".pkl")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
        return sym, pd.read_pickle(path)
    df = None
    if forced != "bitget":
        try:
            df = binance(sym, interval)
        except Exception:
            df = None
    if df is None or len(df) < (2000 if interval == "1h" else 500):
        df = bitget(sym, interval)
    os.makedirs(CACHE, exist_ok=True)
    df.to_pickle(path)
    return sym, df


if __name__ == "__main__":
    for s in WATCHLIST:
        sym, df = load(s)
        print(sym, len(df), df.index[0], df.index[-1])
