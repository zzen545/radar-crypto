"""Mesin backtest 4 jam untuk futures: sinyal di close candle, masuk market di open candle berikutnya.

Biaya 0,08% per sisi (taker futures 0,06% + slippage 0,02%). Bila stop dan target tersentuh
di candle yang sama, dianggap kena stop dulu. Satu posisi per koin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEE = 0.0008


# ---------- indikator ----------
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * pd.Series(ndm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df: pd.DataFrame, n: int, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (arah +1/-1, garis supertrend)."""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = atr(df, n).to_numpy()
    c = df["close"].to_numpy()
    up, dn = hl2 - m * a, hl2 + m * a
    d = np.ones(len(c), dtype=int)
    line = np.full(len(c), np.nan)
    fu, fd = up[0], dn[0]
    for i in range(1, len(c)):
        fu = max(up[i], fu) if c[i - 1] > fu else up[i]
        fd = min(dn[i], fd) if c[i - 1] < fd else dn[i]
        d[i] = 1 if c[i] > fd else -1 if c[i] < fu else d[i - 1]
        line[i] = fu if d[i] == 1 else fd
    return d, line


def prep(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["atr"] = atr(df)
    for n in (9, 12, 20, 21, 26, 50, 55, 100, 200, 300):
        df[f"ema{n}"] = ema(c, n)
    df["rsi2"], df["rsi14"] = rsi(c, 2), rsi(c, 14)
    df["adx"] = adx(df)
    df["sma5"] = c.rolling(5).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    df["bbm"], df["bbu"], df["bbl"] = m, m + 2 * s, m - 2 * s
    df["bbw"] = (4 * s / m)
    macd = ema(c, 12) - ema(c, 26)
    df["macd"], df["macds"] = macd, ema(macd, 9)
    for n in (10, 20, 55):
        df[f"hh{n}"] = df["high"].rolling(n).max().shift(1)
        df[f"ll{n}"] = df["low"].rolling(n).min().shift(1)
    df["vsma"] = df["volume"].rolling(20).mean()
    b = btc["close"].reindex(df.index).ffill()
    df["btc_up"] = np.sign(b - ema(btc["close"], 300).reindex(df.index).ffill())  # ±MA50 harian
    for n, mult in ((10, 3.0), (10, 2.0), (14, 4.0)):
        d, line = supertrend(df, n, mult)
        df[f"st{n}_{mult}"], df[f"stl{n}_{mult}"] = d, line
    return df


# ---------- simulasi ----------
def simulate(df: pd.DataFrame, sig: np.ndarray, stop_dist: np.ndarray, exit_sig: np.ndarray | None,
             tp1: float | None, tp2: float | None, trail: float | None, max_hold: int,
             trail_line: np.ndarray | None = None) -> list[dict]:
    """sig[t] = +1/-1 masuk di open t+1. exit_sig[t] = +1 tutup long / -1 tutup short di open t+1.
    tp1: tutup setengah di tp1 R lalu stop ke entry. tp2: tutup sisa di tp2 R.
    trail: chandelier ATR (dari harga ekstrem). trail_line: garis trailing eksternal (mis. supertrend)."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = df["atr"].to_numpy()
    idx = df.index
    n = len(c)
    out = []
    t = 0
    while t < n - 1:
        d = sig[t]
        if d == 0 or not np.isfinite(stop_dist[t]) or stop_dist[t] <= 0:
            t += 1
            continue
        i0 = t + 1
        entry = o[i0]
        stop = entry - d * stop_dist[t]
        risk = stop_dist[t]
        fee_r = FEE * 2 * entry / risk
        half, realized, best = False, 0.0, entry
        res, nxt = None, None
        for i in range(i0, min(i0 + max_hold, n)):
            adv, fav = (l[i], h[i]) if d == 1 else (h[i], l[i])
            if d * (adv - stop) <= 0:
                px = stop if d * (o[i] - stop) > 0 else o[i]  # gap melewati stop
                res = (i, realized + (0.5 if half else 1) * d * (px - entry) / risk)
                break
            if tp1 and not half and d * (fav - (entry + d * tp1 * risk)) >= 0:
                half, realized = True, 0.5 * tp1
                if d * (entry - stop) > 0:
                    stop = entry
                if not tp2 and not trail and trail_line is None and exit_sig is None:
                    res = (i, tp1)
                    break
            if tp2 and d * (fav - (entry + d * tp2 * risk)) >= 0:
                res = (i, realized + (0.5 if half else 1) * tp2)
                break
            if trail and (half or not tp1):
                best = max(best, h[i]) if d == 1 else min(best, l[i])
                new = best - d * trail * a[i]
                if d * (new - stop) > 0:
                    stop = new
            if trail_line is not None and np.isfinite(trail_line[i]) and d * (trail_line[i] - stop) > 0:
                stop = trail_line[i]
            if exit_sig is not None and exit_sig[i] == d and i + 1 < n:
                res = (i + 1, realized + (0.5 if half else 1) * d * (o[i + 1] - entry) / risk)
                nxt = i  # sinyal balik di candle i boleh langsung jadi entry baru
                break
        if res is None:
            last = min(i0 + max_hold, n) - 1
            if last == n - 1 and i0 + max_hold > n:
                break  # masih terbuka, tidak dihitung
            res = (last, realized + (0.5 if half else 1) * d * (c[last] - entry) / risk)
        ei, r = res
        out.append({"time": idx[t], "side": int(d), "r": r - fee_r, "bars": ei - t, "risk_pct": risk / entry})
        t = ei if nxt is None else nxt
    return out


def summary(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win": 0, "avg": 0, "pf": 0, "total": 0}
    r = np.array([t["r"] for t in trades])
    loss = -r[r < 0].sum()
    return {"n": len(r), "win": round(float((r > 0).mean() * 100), 1), "avg": round(float(r.mean()), 3),
            "pf": round(float(r[r > 0].sum() / loss), 2) if loss else 99, "total": round(float(r.sum()), 1)}


def compound(trades: list[dict], risk: float = 0.05, start: float = 1000) -> tuple[float, float]:
    eq, peak, dd = start, start, 0.0
    for t in trades:
        eq *= 1 + risk * t["r"]
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    return eq, dd
