"""Eksperimen cara masuk x cara keluar untuk menaikkan win rate tanpa merusak profit.

Aturan dipilih di data 2019-2023 (in-sample), lalu dicek di 2024-2026 (out-of-sample).
Sinyal arah tetap sama dengan dashboard: skor teknikal + volume, filter BTC di atas/bawah MA50.

    python eksperimen.py            # semua kombinasi, ringkasan per periode
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

import backtest as bt
from analyzer import indicators, signals

SPLIT = "2024-01-01"
MIN_WARMUP = 60


def prepare(symbol: str) -> tuple[str, pd.DataFrame]:
    """Hitung sekali per koin: indikator, skor harian, sisi (LONG/SHORT/0) setelah filter BTC."""
    sym, _, raw = bt.load(symbol)
    df = indicators.add_all(raw).join(bt.btc_context(indicators.add_all(raw)))
    df["atr"] = bt.atr(df)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["hh20"] = df["high"].rolling(20).max().shift(1)
    df["ll20"] = df["low"].rolling(20).min().shift(1)
    sign = np.sign(df["close"].diff()).fillna(0)
    obv = (sign * df["volume"]).cumsum()
    obv_sma = obv.rolling(20).mean()
    side = np.zeros(len(df), dtype=int)
    score = np.zeros(len(df), dtype=int)
    for t in range(MIN_WARMUP, len(df)):
        s = signals.analyze(sym, df.iloc[:t + 1]).score + bt.volume_score(df, obv, obv_sma, t)
        score[t] = s
        d = 1 if s >= 3 else -1 if s <= -3 else 0
        if d and np.sign(df["btc"].iloc[t] - df["btc_sma50"].iloc[t]) == -d:
            d = 0
        side[t] = d
    df["score"], df["side"] = score, side
    return sym, df


def entry_order(df: pd.DataFrame, t: int, d: int, mode: str) -> tuple[float | None, float, int] | None:
    """(harga limit atau None=market di open berikutnya, stop, berapa candle order berlaku)."""
    c, a = df["close"].iloc[t], df["atr"].iloc[t]
    if mode == "sinyal":  # seperti dashboard sekarang
        sup, res = bt.levels(df, t)
        p = bt.make_plan("LONG" if d == 1 else "SHORT", c, a, sup, res)
        return p["entry"], p["stop"], 3
    if mode == "pullback":  # tunggu harga kembali ke EMA20 searah tren
        e = df["ema20"].iloc[t]
        if d * (c - e) <= 0:
            return None
        return e, e - d * 1.5 * a, 7
    if mode == "breakout":  # hanya saat close menembus high/low 20 hari
        lvl = df["hh20"].iloc[t] if d == 1 else df["ll20"].iloc[t]
        if not d * (c - lvl) > 0:
            return None
        return None, c - d * 2 * a, 1
    raise ValueError(mode)


def run_trade(df: pd.DataFrame, t: int, d: int, order, exit_mode: str):
    """Simulasi satu trade. Kembalikan (idx exit, R bersih) atau None bila order tidak terisi."""
    limit, stop, valid = order
    o, hi, lo, cl, atr = (df[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
    n = len(df)
    fill, entry = None, None
    for i in range(t + 1, min(t + 1 + valid, n)):
        if limit is None:
            fill, entry = i, o[i]
            break
        if (lo[i] <= limit) if d == 1 else (hi[i] >= limit):
            fill, entry = i, (min(o[i], limit) if d == 1 else max(o[i], limit))
            break
    if fill is None:
        return None
    risk = d * (entry - stop)
    if risk <= 0:
        return None
    fee_r = bt.FEE * 2 * entry / risk
    tp1_r, tp2_r, trail, max_hold = {
        "1.5R/3R": (1.5, 3.0, None, 30),
        "1R/2R": (1.0, 2.0, None, 30),
        "1R/trail": (1.0, None, 3.0, 60),
        "1.5R/trail": (1.5, None, 3.0, 60),
        "trail": (None, None, 3.0, 90),
    }[exit_mode]
    half, realized, best = False, 0.0, entry
    for i in range(fill, min(fill + max_hold, n)):
        adverse, favor = (lo[i], hi[i]) if d == 1 else (hi[i], lo[i])
        if d * (adverse - stop) <= 0:
            rest = 0.5 if half else 1.0
            return i, realized + rest * d * (stop - entry) / risk - fee_r
        if tp1_r and not half and d * (favor - (entry + d * tp1_r * risk)) >= 0:
            half, realized = True, 0.5 * tp1_r
            stop = entry if d * (entry - stop) > 0 else stop
        if tp2_r and half and d * (favor - (entry + d * tp2_r * risk)) >= 0:
            return i, realized + 0.5 * tp2_r - fee_r
        if trail and (half or not tp1_r):
            best = max(best, hi[i]) if d == 1 else min(best, lo[i])
            new = best - d * trail * atr[i]
            if d * (new - stop) > 0:
                stop = new
        if i == n - 1:
            return None
    last = min(fill + max_hold, n) - 1
    rest = 0.5 if half else 1.0
    return last, realized + rest * d * (cl[last] - entry) / risk - fee_r


def run(data: dict, entry_mode: str, exit_mode: str) -> list[dict]:
    trades = []
    for sym, df in data.items():
        t = MIN_WARMUP if len(df) < bt.WARMUP + 50 else bt.WARMUP
        side = df["side"].to_numpy()
        while t < len(df) - 1:
            d = int(side[t])
            order = entry_order(df, t, d, entry_mode) if d else None
            out = run_trade(df, t, d, order, exit_mode) if order else None
            if out is None:
                t += 1
                continue
            exit_i, r = out
            trades.append({"symbol": sym, "side": d, "date": str(df.index[t].date()), "r": r,
                           "days": exit_i - t})
            t = exit_i + 1
    return trades


def summary(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    r = np.array([t["r"] for t in trades])
    loss = -r[r < 0].sum()
    return {"n": len(r), "win": round((r > 0).mean() * 100, 1), "avg": round(r.mean(), 3),
            "pf": round(r[r > 0].sum() / loss, 2) if loss else None, "total": round(r.sum(), 1),
            "days": round(float(np.mean([t["days"] for t in trades])), 1)}


def main() -> None:
    data = {}
    for s in bt.WATCHLIST:
        try:
            sym, df = prepare(s)
            data[sym] = df
        except Exception as exc:
            print(f"[!] {s}: {exc}", file=sys.stderr)
    out = {}
    for em, xm in itertools.product(["sinyal", "pullback", "breakout"], ["1.5R/3R", "1R/2R", "1R/trail", "1.5R/trail", "trail"]):
        tr = run(data, em, xm)
        ins = summary([t for t in tr if t["date"] < SPLIT])
        oos = summary([t for t in tr if t["date"] >= SPLIT])
        out[f"{em} | {xm}"] = {"in": ins, "out": oos}
        print(f"{em:9} {xm:10}  IN  n{ins['n']:4} win {ins['win']:5}% pf {ins['pf']} avg {ins['avg']:+.3f}   "
              f"OUT n{oos['n']:4} win {oos['win']:5}% pf {oos['pf']} avg {oos['avg']:+.3f} hari {oos['days']}")
    json.dump(out, open("hasil/eksperimen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
