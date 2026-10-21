"""Sumber data OHLCV.

Setiap sumber data mengembalikan DataFrame dengan index waktu (UTC) dan kolom:
open, high, low, close, volume. Untuk menambah sumber baru (mis. saham IDX),
buat class turunan DataSource dan daftarkan di SOURCES.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSource:
    name = "base"

    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError

    def normalize_symbol(self, symbol: str) -> str:
        return symbol


class BinanceSource(DataSource):
    """Data publik Binance, tanpa API key."""

    name = "binance"
    BASE_URLS = ("https://api.binance.com", "https://data-api.binance.vision")
    INTERVALS = {"15m", "1h", "4h", "1d", "1w"}

    def normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().replace("/", "").replace("-", "")
        quotes = ("USDT", "FDUSD", "BTC", "ETH", "IDR")
        has_quote = any(symbol.endswith(q) and len(symbol) > len(q) for q in quotes)
        return symbol if has_quote else symbol + "USDT"

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 300) -> pd.DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"Interval {interval!r} tidak didukung. Pilih: {sorted(self.INTERVALS)}")
        params = {"symbol": self.normalize_symbol(symbol), "interval": interval, "limit": min(limit, 1000)}
        errors = []
        for base in self.BASE_URLS:
            try:
                resp = requests.get(f"{base}/api/v3/klines", params=params, timeout=15)
                if resp.status_code == 451:
                    raise ConnectionError("Binance memblokir lokasi/IP ini (HTTP 451)")
                resp.raise_for_status()
                rows = resp.json()
                break
            except (requests.RequestException, ConnectionError) as exc:
                errors.append(f"{base}: {exc}")
        else:
            raise ConnectionError(f"Gagal mengambil data {params['symbol']} dari Binance:\n    " + "\n    ".join(errors))

        df = pd.DataFrame(rows).iloc[:, :7]
        df.columns = ["time"] + COLUMNS + ["close_time"]
        # Candle terakhir biasanya masih berjalan: buang agar volume/RSI tidak menyesatkan
        now_ms = pd.Timestamp.now(tz="UTC").value // 1_000_000
        df = df[df["close_time"] < now_ms].drop(columns="close_time")
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.set_index("time").astype(float)
        return df


class BitgetSource(DataSource):
    """Data publik Bitget, tanpa API key. Coba pasar spot dulu, lalu futures USDT-M
    (mis. XAUUSDT emas hanya ada di futures)."""

    name = "bitget"
    BASE_URL = "https://api.bitget.com"
    # interval -> (granularity spot, granularity futures, durasi candle)
    INTERVALS = {
        "15m": ("15min", "15m", pd.Timedelta(minutes=15)),
        "1h": ("1h", "1H", pd.Timedelta(hours=1)),
        "4h": ("4h", "4H", pd.Timedelta(hours=4)),
        "1d": ("1day", "1D", pd.Timedelta(days=1)),
        "1w": ("1week", "1W", pd.Timedelta(weeks=1)),
    }

    def normalize_symbol(self, symbol: str) -> str:
        return BinanceSource().normalize_symbol(symbol)

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 300) -> pd.DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"Interval {interval!r} tidak didukung. Pilih: {sorted(self.INTERVALS)}")
        spot_gran, fut_gran, step = self.INTERVALS[interval]
        sym = self.normalize_symbol(symbol)
        limit = min(limit, 1000)
        markets = [
            ("spot", "/api/v2/spot/market/", {"symbol": sym, "granularity": spot_gran}),
            ("futures", "/api/v2/mix/market/",
             {"symbol": sym, "productType": "USDT-FUTURES", "granularity": fut_gran}),
        ]
        errors = []
        for market, path, params in markets:
            try:
                rows = self._get(path + "candles", {**params, "limit": limit})
                break
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{market}: {exc}")
        else:
            raise ConnectionError(f"Gagal mengambil data {sym} dari Bitget:\n    " + "\n    ".join(errors))

        # Satu request hanya memberi 90-300 candle; sisanya diambil mundur lewat history-candles
        while len(rows) < limit:
            oldest = min(int(r[0]) for r in rows)
            try:
                older = self._get(path + "history-candles", {**params, "endTime": oldest, "limit": 200})
            except (requests.RequestException, ValueError):
                break
            older = [r for r in older if int(r[0]) < oldest]
            if not older:
                break
            rows = older + rows

        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["time"] + COLUMNS
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df = df.set_index("time").astype(float).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # Candle terakhir yang masih berjalan dibuang, sama seperti BinanceSource
        return df[df.index + step <= pd.Timestamp.now(tz="UTC")].tail(limit)

    def _get(self, path: str, params: dict) -> list:
        resp = requests.get(self.BASE_URL + path, params=params, timeout=15)
        body = resp.json()
        if resp.status_code != 200 or body.get("code") != "00000" or not body.get("data"):
            raise ValueError(body.get("msg") or f"HTTP {resp.status_code}")
        return body["data"]


class AutoSource(DataSource):
    """Binance dulu; kalau koin tidak ada atau datanya kurang, ambil dari Bitget.

    Awali simbol dengan `bitget:` (mis. `bitget:LIT`) untuk memaksa Bitget, berguna bila
    ticker yang sama di Binance ternyata koin lain. Sumber yang dipakai dicatat di
    `df.attrs["source"]`.
    """

    name = "auto"
    MIN_CANDLES = 60

    def __init__(self) -> None:
        self.primary, self.fallback = BinanceSource(), BitgetSource()

    def _split(self, symbol: str) -> tuple[str | None, str]:
        prefix, sep, rest = symbol.partition(":")
        return (prefix.lower(), rest) if sep else (None, symbol)

    def normalize_symbol(self, symbol: str) -> str:
        return self.primary.normalize_symbol(self._split(symbol)[1])

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 300) -> pd.DataFrame:
        forced, sym = self._split(symbol)
        if forced is not None and forced not in ("binance", "bitget"):
            raise ValueError(f"Prefix {forced!r} tidak dikenal. Pakai binance: atau bitget:")
        order = {"binance": [self.primary], "bitget": [self.fallback]}.get(forced, [self.primary, self.fallback])
        errors, best = [], None
        for src in order:
            try:
                df = src.fetch(sym, interval, limit)
            except Exception as exc:
                errors.append(str(exc))
                continue
            df.attrs["source"] = src.name
            if len(df) >= self.MIN_CANDLES:
                return df
            if best is None or len(df) > len(best):
                best = df
        if best is not None:
            return best  # main.py yang memutuskan datanya cukup atau tidak
        raise ConnectionError("\n".join(errors))


class CSVSource(DataSource):
    """Baca dari file CSV (kolom: time/date, open, high, low, close, volume).

    Berguna untuk data offline atau data saham hasil ekspor.
    `symbol` diartikan sebagai path file CSV.
    """

    name = "csv"

    def normalize_symbol(self, symbol: str) -> str:
        return Path(symbol).stem.upper()

    def fetch(self, symbol: str, interval: str = "1d", limit: int = 300) -> pd.DataFrame:
        df = pd.read_csv(symbol)
        df.columns = [c.strip().lower() for c in df.columns]
        time_col = next((c for c in ("time", "date", "datetime", "timestamp") if c in df.columns), None)
        if time_col is None:
            raise ValueError("CSV harus punya kolom time/date")
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col)[COLUMNS].astype(float).sort_index()
        return df.tail(limit)


SOURCES = {cls.name: cls for cls in (AutoSource, BinanceSource, BitgetSource, CSVSource)}


def get_source(name: str) -> DataSource:
    try:
        return SOURCES[name]()
    except KeyError:
        raise ValueError(f"Sumber data {name!r} tidak dikenal. Pilih: {sorted(SOURCES)}") from None
