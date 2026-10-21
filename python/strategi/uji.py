"""Uji banyak strategi 4 jam. Pilih di 2022-2024 (in-sample), cek di 2025-2026 (out-of-sample)."""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

import os

import data4h
import mesin as m

IV = os.environ.get("IV", "4h")
HOLD = {"4h": 1, "1h": 4, "1d": 1}[IV]
IS_START, SPLIT = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")


def cross(a, b):
    a, b = np.asarray(a), np.asarray(b)
    up = (a > b) & (np.roll(a, 1) <= np.roll(b, 1))
    dn = (a < b) & (np.roll(a, 1) >= np.roll(b, 1))
    up[0] = dn[0] = False
    return up, dn


def strategies(df):
    """Yield (nama, sig, stop_dist, exit_sig, trail_line, exit_mode_cocok)."""
    c, a = df["close"].to_numpy(), df["atr"].to_numpy()
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    z = np.zeros(len(c), dtype=int)

    # 1. Supertrend flip, keluar saat flip balik (stop = garis supertrend)
    for key in ("10_3.0", "10_2.0", "14_4.0"):
        d = df[f"st{key}"].to_numpy(); line = df[f"stl{key}"].to_numpy()
        sig = np.where(d != np.roll(d, 1), d, 0); sig[0] = 0
        yield f"supertrend {key}", sig, np.abs(c - line), -sig, line, "own"

    # 2. EMA cross, keluar cross balik, stop 2 ATR
    for f, s in ((9, 21), (20, 50), (21, 55), (50, 200)):
        up, dn = cross(df[f"ema{f}"], df[f"ema{s}"])
        sig = np.where(up, 1, np.where(dn, -1, 0))
        yield f"ema {f}/{s}", sig, 2 * a, -sig, None, "own"
        yield f"ema {f}/{s}", sig, 2 * a, None, None, "tp"

    # 3. Donchian breakout
    for nn, ex in ((20, 10), (55, 20)):
        sig = np.where(c > df[f"hh{nn}"].to_numpy(), 1, np.where(c < df[f"ll{nn}"].to_numpy(), -1, 0))
        exs = np.where(c < df[f"ll{ex}"].to_numpy(), 1, np.where(c > df[f"hh{ex}"].to_numpy(), -1, 0))
        yield f"donchian {nn}/{ex}", sig, 2 * a, exs, None, "own"
        yield f"donchian {nn}/{ex}", sig, 2 * a, None, None, "tp"

    # 4. Pullback ke EMA20 searah tren (EMA50 > EMA200), candle menyentuh EMA20 lalu close di atasnya
    e20, e50, e200 = (df[f"ema{n}"].to_numpy() for n in (20, 50, 200))
    lsig = (e50 > e200) & (l <= e20) & (c > e20) & (c > e50)
    ssig = (e50 < e200) & (h >= e20) & (c < e20) & (c < e50)
    sig = np.where(lsig, 1, np.where(ssig, -1, 0))
    yield "pullback ema20", sig, 1.5 * a, None, None, "tp"

    # 5. Mean reversion RSI(2) searah tren EMA200, keluar saat close lewat SMA5
    r2, s5 = df["rsi2"].to_numpy(), df["sma5"].to_numpy()
    for lo in (5, 10):
        sig = np.where((r2 < lo) & (c > e200), 1, np.where((r2 > 100 - lo) & (c < e200), -1, 0))
        exs = np.where(c > s5, 1, np.where(c < s5, -1, 0))
        yield f"rsi2<{lo} reversion", sig, 2.5 * a, exs, None, "own"
        yield f"rsi2<{lo} reversion", sig, 1.5 * a, None, None, "tp"

    # 6. Bollinger: kembali ke tengah searah tren
    bbl, bbu, bbm = (df[k].to_numpy() for k in ("bbl", "bbu", "bbm"))
    sig = np.where((l < bbl) & (c > bbl) & (c > e200), 1, np.where((h > bbu) & (c < bbu) & (c < e200), -1, 0))
    exs = np.where(c > bbm, 1, np.where(c < bbm, -1, 0))
    yield "bollinger reversion", sig, 2 * a, exs, None, "own"
    yield "bollinger reversion", sig, 1.5 * a, None, None, "tp"

    # 7. Bollinger squeeze breakout
    bbw = df["bbw"]; sq = (bbw < bbw.rolling(120).quantile(0.2)).shift(1).fillna(False).to_numpy()
    sig = np.where(sq & (c > bbu), 1, np.where(sq & (c < bbl), -1, 0))
    yield "squeeze breakout", sig, 2 * a, None, None, "tp"

    # 8. MACD cross searah EMA200
    up, dn = cross(df["macd"], df["macds"])
    sig = np.where(up & (c > e200), 1, np.where(dn & (c < e200), -1, 0))
    yield "macd cross", sig, 2 * a, -sig, None, "own"
    yield "macd cross", sig, 2 * a, None, None, "tp"


EXITS_TP = {"1R": (1, None, None, 30), "1R/2R": (1, 2, None, 60), "1.5R/3R": (1.5, 3, None, 90),
            "2R": (None, 2, None, 60), "1R/trail3": (1, None, 3, 120), "trail3": (None, None, 3, 180)}
FILTERS = ("none", "btc", "htf", "btc+htf", "adx")


def apply_filter(df, sig, f):
    s = sig.copy()
    if "btc" in f:
        s[s != df["btc_up"].to_numpy()] = 0
    if "htf" in f:
        htf = np.sign(df["close"] - df["ema300"]).to_numpy()
        s[s != htf] = 0
    if f == "adx":
        s[df["adx"].to_numpy() < 20] = 0
    return s


def main():
    _, btc = data4h.load("BTC", IV)
    data = {}
    for s in data4h.WATCHLIST:
        sym, df = data4h.load(s, IV)
        data[sym] = m.prep(df, btc)
    results = {}
    trades_all = {}
    for sym, df in data.items():
        for name, sig, sd, exs, line, kind in strategies(df):
            exits = {"own": (None, None, None, 180)} if kind == "own" else EXITS_TP
            for xn, (t1, t2, tr, mh) in exits.items():
                for f in FILTERS:
                    key = f"{name} | {xn} | {f}"
                    tr_ = m.simulate(df, apply_filter(df, sig, f), sd, exs, t1, t2, tr, mh * HOLD, line)
                    trades_all.setdefault(key, []).extend({**t, "sym": sym} for t in tr_)
    rows = []
    for key, tr in trades_all.items():
        ins = [t for t in tr if IS_START <= t["time"] < SPLIT]
        oos = [t for t in tr if t["time"] >= SPLIT]
        y26 = [t for t in tr if t["time"].year == 2026]
        rows.append({"key": key, "in": m.summary(ins), "out": m.summary(oos), "2026": m.summary(y26)})
    pd.to_pickle(trades_all, f"../hasil/cache4h/trades_{IV}.pkl")
    json.dump(rows, open("../hasil/uji4h.json" if IV == "4h" else f"../hasil/uji_{IV}.json", "w"), indent=0, default=str)
    score = lambda r: r["in"]["avg"] * np.sqrt(r["in"]["n"]) if r["in"]["n"] >= 100 else -9
    rows.sort(key=score, reverse=True)
    print(f"{'strategi':45} {'IN n':>5} {'win':>5} {'avg':>6} {'pf':>5} | {'OUT n':>5} {'win':>5} {'avg':>6} {'pf':>5} | 2026 n  avg")
    for r in rows[:40]:
        i, o, y = r["in"], r["out"], r["2026"]
        print(f"{r['key']:45} {i['n']:5} {i['win']:5} {i['avg']:+6.3f} {i['pf']:5} | {o['n']:5} {o['win']:5} {o['avg']:+6.3f} {o['pf']:5} | {y['n']:4} {y['avg']:+.3f}")


if __name__ == "__main__":
    main()
