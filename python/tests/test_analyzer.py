"""Uji logika indikator dan sinyal dengan data buatan (tanpa internet)."""
import numpy as np
import pandas as pd

from analyzer import indicators, signals


def make_df(drift: float, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.uniform(900, 1100, n)}, index=idx)


def test_rsi_bounds_and_trend():
    up = indicators.rsi(make_df(0.01)["close"]).dropna()
    down = indicators.rsi(make_df(-0.01)["close"]).dropna()
    assert up.between(0, 100).all() and down.between(0, 100).all()
    assert up.iloc[-1] > 60 and down.iloc[-1] < 40


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 50, dtype=float))
    assert indicators.rsi(s).iloc[-1] == 100


def test_bollinger_contains_mid():
    bb = indicators.bollinger(make_df(0)["close"]).dropna()
    assert (bb["bb_lower"] <= bb["bb_mid"]).all() and (bb["bb_mid"] <= bb["bb_upper"]).all()


def test_trend_classification():
    assert signals.analyze("UP", indicators.add_all(make_df(0.01))).trend == "BULLISH"
    assert signals.analyze("DOWN", indicators.add_all(make_df(-0.01))).trend == "BEARISH"


def test_chart_writes_png(tmp_path):
    from analyzer import chart
    df = indicators.add_all(make_df(0.005))
    path = chart.plot("TEST", df, "BULLISH", tmp_path)
    assert path.exists() and path.stat().st_size > 10_000


def test_binance_symbol_normalization():
    from analyzer.data_sources import BinanceSource
    n = BinanceSource().normalize_symbol
    assert n("BTC") == "BTCUSDT" and n("eth") == "ETHUSDT"
    assert n("ETHBTC") == "ETHBTC" and n("sol/usdt") == "SOLUSDT"


def test_bitget_parses_and_falls_back_to_futures(monkeypatch):
    from analyzer import data_sources

    now = pd.Timestamp.now(tz="UTC").floor("D")
    times = [now - pd.Timedelta(days=d) for d in (0, 1, 2)]  # urutan terbalik; hari ini belum selesai
    rows = [[str(t.value // 1_000_000), "1", "2", "0.5", "1.5", "10", "15"] for t in times]
    calls = []

    class Resp:
        def __init__(self, body, status=200):
            self.body, self.status_code = body, status

        def json(self):
            return self.body

    def fake_get(url, params, timeout):
        calls.append(url)
        if "/spot/" in url:
            return Resp({"code": "40034", "msg": "Parameter does not exist"}, 400)
        if "history-candles" in url:  # halaman lebih lama: kosong
            return Resp({"code": "00000", "msg": "success", "data": []})
        return Resp({"code": "00000", "msg": "success", "data": rows})

    monkeypatch.setattr(data_sources.requests, "get", fake_get)
    df = data_sources.get_source("bitget").fetch("XAU", "1d", 10)
    assert "/spot/" in calls[0] and calls[1].endswith("/mix/market/candles")
    assert list(df.columns) == data_sources.COLUMNS
    assert len(df) == 2 and df.index.is_monotonic_increasing


def test_auto_falls_back_to_bitget(monkeypatch):
    from analyzer import data_sources

    long_df = make_df(0.0, n=100)
    short_df = make_df(0.0, n=20)
    calls = []

    def binance(self, symbol, interval, limit):
        calls.append(("binance", symbol))
        if symbol == "NOPE":
            raise ConnectionError("Invalid symbol")
        return short_df.copy() if symbol == "NEW" else long_df.copy()

    def bitget(self, symbol, interval, limit):
        calls.append(("bitget", symbol))
        return long_df.copy()

    monkeypatch.setattr(data_sources.BinanceSource, "fetch", binance)
    monkeypatch.setattr(data_sources.BitgetSource, "fetch", bitget)
    src = data_sources.get_source("auto")
    assert src.fetch("BTC").attrs["source"] == "binance"
    assert src.fetch("NOPE").attrs["source"] == "bitget"      # tidak ada di Binance
    assert src.fetch("NEW").attrs["source"] == "bitget"       # data Binance kurang
    calls.clear()
    assert src.fetch("bitget:LIT").attrs["source"] == "bitget"
    assert calls == [("bitget", "LIT")]
    assert src.normalize_symbol("bitget:lit") == "LITUSDT"
