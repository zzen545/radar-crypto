"""Price action / SMC: struktur tren mayor (harian) & minor (4 jam), Fibonacci, pola candle,
liquidity sweep, fair value gap. Semua deteksi hanya memakai candle yang sudah close (tanpa melihat masa depan).

Sinyal dibuat di close candle t. Order: market di open t+1, atau limit yang berlaku beberapa candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mesin as m

FEE = m.FEE


# ---------- struktur ----------
def swings(h: np.ndarray, l: np.ndarray, k: int):
    """Swing high/low fraktal k candle kiri-kanan. Swing di i baru 'diketahui' di i+k."""
    n = len(h)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        win_h, win_l = h[i - k:i + k + 1], l[i - k:i + k + 1]
        sh[i] = h[i] == win_h.max() and (win_h == h[i]).sum() == 1
        sl[i] = l[i] == win_l.min() and (win_l == l[i]).sum() == 1
    return sh, sl


_SCACHE: dict = {}
_CC: dict = {}


def structure(df: pd.DataFrame, k: int) -> pd.DataFrame:
    key = (id(df), len(df), k)
    if key not in _SCACHE:
        _SCACHE[key] = _structure(df, k)
    return _SCACHE[key]


def _structure(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per candle t (yang sudah close): swing high/low terakhir yang sudah terkonfirmasi, sebelumnya,
    dan arah struktur (+1 setelah close menembus swing high / BOS naik, -1 setelah menembus swing low)."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    n = len(c)
    sh, sl = swings(h, l, k)
    last_h = np.full(n, np.nan); last_l = np.full(n, np.nan)
    last_hi = np.full(n, -1); last_li = np.full(n, -1)
    prev_h = np.full(n, np.nan); prev_l = np.full(n, np.nan)
    trend = np.zeros(n, int)
    lh = ll = ph = pl = np.nan; lhi = lli = -1; tr = 0
    broken_h = broken_l = False
    for t in range(n):
        j = t - k  # swing di j terkonfirmasi sekarang
        if j >= 0 and sh[j]:
            ph, lh, lhi, broken_h = lh, h[j], j, False
        if j >= 0 and sl[j]:
            pl, ll, lli, broken_l = ll, l[j], j, False
        if np.isfinite(lh) and not broken_h and c[t] > lh:
            tr, broken_h = 1, True
        if np.isfinite(ll) and not broken_l and c[t] < ll:
            tr, broken_l = -1, True
        last_h[t], last_l[t], last_hi[t], last_li[t] = lh, ll, lhi, lli
        prev_h[t], prev_l[t], trend[t] = ph, pl, tr
    return pd.DataFrame({"sh": last_h, "sl": last_l, "shi": last_hi, "sli": last_li,
                         "psh": prev_h, "psl": prev_l, "trend": trend}, index=df.index)


def major_trend(df4: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Tren mayor dari candle harian (dibentuk dari 4 jam), dipakai hanya setelah candle harian close."""
    d = df4.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    st = structure(d, k)["trend"]
    st.index = st.index + pd.Timedelta(days=1)  # baru diketahui setelah hari itu selesai
    return st.reindex(df4.index, method="ffill").fillna(0).to_numpy().astype(int)


# ---------- pola candle ----------
def candles(df: pd.DataFrame):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    body = np.abs(c - o); rng = np.maximum(h - l, 1e-12)
    po, pc = np.roll(o, 1), np.roll(c, 1)
    bull_eng = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear_eng = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    low_wick = np.minimum(o, c) - l; up_wick = h - np.maximum(o, c)
    bull_pin = (low_wick >= 2 * body) & (low_wick >= 0.6 * rng) & (c > l + 0.5 * rng)
    bear_pin = (up_wick >= 2 * body) & (up_wick >= 0.6 * rng) & (c < h - 0.5 * rng)
    bull_eng[0] = bear_eng[0] = False
    return bull_eng | bull_pin, bear_eng | bear_pin


# ---------- setup ----------
def setups(df: pd.DataFrame, k: int = 3, kind: str = "fib", need_major: bool = True, need_minor: bool = False,
           need_pattern: bool = True, fib=(0.5, 0.786), tp_mode: str = "struct"):
    """Kembalikan list order: (t, arah, limit|None, stop, target, berlaku_candle)."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = df["atr"].to_numpy()
    st = structure(df, k)
    shp, slp, shi, sli = st["sh"].to_numpy(), st["sl"].to_numpy(), st["shi"].to_numpy(), st["sli"].to_numpy()
    minor = st["trend"].to_numpy()
    major = df["major"].to_numpy()
    bull_c, bear_c = candles(df)
    n = len(c)
    orders = []
    for t in range(60, n - 1):
        for d in (1, -1):
            if need_major and major[t] != d:
                continue
            if need_minor and minor[t] != d:
                continue
            pat = bull_c[t] if d == 1 else bear_c[t]
            if need_pattern and not pat:
                continue
            buf = 0.2 * a[t]
            if kind == "fib":
                # kaki impuls terakhir: swing low -> swing high (long), keduanya sudah terkonfirmasi
                if not (np.isfinite(shp[t]) and np.isfinite(slp[t])):
                    continue
                if d == 1 and not sli[t] < shi[t]:
                    continue
                if d == -1 and not shi[t] < sli[t]:
                    continue
                lo, hi = slp[t], shp[t]
                leg = hi - lo
                if leg < 2 * a[t]:
                    continue
                # harga sekarang di zona retracement dan belum menembus ujung kaki
                if d == 1:
                    z_top, z_bot = hi - fib[0] * leg, hi - fib[1] * leg
                    if not (l[t] <= z_top and c[t] >= z_bot and l[t] > lo):
                        continue
                    stop, target = lo - buf, (hi if tp_mode == "struct" else hi + 0.272 * leg)
                else:
                    z_bot, z_top = lo + fib[0] * leg, lo + fib[1] * leg
                    if not (h[t] >= z_bot and c[t] <= z_top and h[t] < hi):
                        continue
                    stop, target = hi + buf, (lo if tp_mode == "struct" else lo - 0.272 * leg)
                orders.append((t, d, None, stop, target, 1))
            elif kind == "sweep":
                # liquidity sweep: ekor menembus swing terakhir, close kembali di dalam
                if d == 1:
                    if not (np.isfinite(slp[t]) and l[t] < slp[t] and c[t] > slp[t]):
                        continue
                    stop, target = l[t] - buf, shp[t]
                else:
                    if not (np.isfinite(shp[t]) and h[t] > shp[t] and c[t] < shp[t]):
                        continue
                    stop, target = h[t] + buf, slp[t]
                if not np.isfinite(target) or d * (target - c[t]) <= 0:
                    target = np.nan
                orders.append((t, d, None, stop, target, 1))
    return orders


def fvg_orders(df: pd.DataFrame, k: int = 3, need_major=True, need_minor=False, valid: int = 30, min_gap=0.3):
    """Fair value gap searah tren: gap 3 candle (low[i] > high[i-2] untuk bullish), limit di tepi gap,
    stop di bawah candle i-2 / dasar gap, target swing high terakhir."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = df["atr"].to_numpy()
    st = structure(df, k)
    shp, slp, minor = st["sh"].to_numpy(), st["sl"].to_numpy(), st["trend"].to_numpy()
    major = df["major"].to_numpy()
    orders = []
    for t in range(60, len(c) - 1):
        for d in (1, -1):
            if need_major and major[t] != d:
                continue
            if need_minor and minor[t] != d:
                continue
            if d == 1 and l[t] - h[t - 2] >= min_gap * a[t] and c[t - 1] > o[t - 1]:
                top, bot = l[t], h[t - 2]
                orders.append((t, 1, top, bot - 0.5 * (top - bot) - 0.1 * a[t], shp[t] if shp[t] > top else np.nan, valid))
            if d == -1 and l[t - 2] - h[t] >= min_gap * a[t] and c[t - 1] < o[t - 1]:
                bot, top = h[t], l[t - 2]
                orders.append((t, -1, bot, top + 0.5 * (top - bot) + 0.1 * a[t], slp[t] if slp[t] < bot else np.nan, valid))
    return orders


# ---------- simulasi ----------
def simulate(df: pd.DataFrame, orders, rr_min: float = 1.0, rr_default: float = 2.0, rr_cap: float = 6.0,
             be_at: float | None = None, max_hold: int = 120):
    """Satu posisi per koin. Target struktural dipakai bila RR-nya >= rr_min (maks rr_cap); kalau tidak ada,
    pakai rr_default. be_at: geser stop ke entry setelah untung be_at R."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    idx, n = df.index, len(c)
    out, busy_until = [], -1
    for (t, d, limit, stop, target, valid) in orders:
        if t <= busy_until:
            continue
        fill = None
        for i in range(t + 1, min(t + 1 + valid, n)):
            if limit is None:
                fill, entry = i, o[i]; break
            if (l[i] <= limit) if d == 1 else (h[i] >= limit):
                fill, entry = i, (min(o[i], limit) if d == 1 else max(o[i], limit)); break
            if d * (c[i] - stop) <= 0:
                break  # setup batal: close menembus stop sebelum terisi
        if fill is None:
            continue
        risk = d * (entry - stop)
        if risk <= 0:
            continue
        rr = d * (target - entry) / risk if np.isfinite(target) else np.nan
        if not np.isfinite(rr) or rr < rr_min:
            rr = rr_default if not np.isfinite(rr) else None
        if rr is None:
            continue
        rr = min(rr, rr_cap)
        tp = entry + d * rr * risk
        cur_stop, res = stop, None
        for i in range(fill, min(fill + max_hold, n)):
            adv, fav = (l[i], h[i]) if d == 1 else (h[i], l[i])
            if d * (adv - cur_stop) <= 0:
                px = cur_stop if d * (o[i] - cur_stop) > 0 else o[i]
                res = (i, d * (px - entry) / risk); break
            if d * (fav - tp) >= 0:
                res = (i, rr); break
            if be_at and d * (fav - (entry + d * be_at * risk)) >= 0 and d * (entry - cur_stop) > 0:
                cur_stop = entry
        if res is None:
            last = min(fill + max_hold, n) - 1
            if fill + max_hold > n:
                continue  # masih terbuka
            res = (last, d * (c[last] - entry) / risk)
        ei, r = res
        out.append({"time": idx[t], "side": d, "r": r - FEE * 2 * entry / risk, "bars": ei - t, "rr": rr,
                    "risk_pct": risk / entry})
        busy_until = ei
    return out


def smc_orders(df: pd.DataFrame, k: int = 3, sweep_look: int = 12, need_major=True, need_minor=False,
               discount: bool = True, min_gap: float = 0.3, valid: int = 20):
    """Model SMC: liquidity sweep swing low (long) -> displacement membentuk FVG bullish dalam
    `sweep_look` candle -> limit di tepi atas FVG. Opsional FVG harus di zona diskon (< 50% range sweep..swing high).
    Stop di bawah low sweep, target swing high terakhir (likuiditas di atas)."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = df["atr"].to_numpy()
    st = structure(df, k)
    shp, slp, minor = st["sh"].to_numpy(), st["sl"].to_numpy(), st["trend"].to_numpy()
    major = df["major"].to_numpy()
    n = len(c)
    last_sweep = {1: (-10**9, np.nan, np.nan), -1: (-10**9, np.nan, np.nan)}  # (t, ekstrem sweep, swing lawan)
    orders = []
    for t in range(60, n - 1):
        if np.isfinite(slp[t]) and l[t] < slp[t] and c[t] > slp[t]:
            last_sweep[1] = (t, l[t], shp[t])
        if np.isfinite(shp[t]) and h[t] > shp[t] and c[t] < shp[t]:
            last_sweep[-1] = (t, h[t], slp[t])
        for d in (1, -1):
            if need_major and major[t] != d:
                continue
            if need_minor and minor[t] != d:
                continue
            ts, ext, opp = last_sweep[d]
            if t - ts > sweep_look or t - ts < 1:
                continue
            ext = min(ext, l[ts:t + 1].min()) if d == 1 else max(ext, h[ts:t + 1].max())
            if d == 1 and l[t] - h[t - 2] >= min_gap * a[t]:
                top = l[t]
                if discount and np.isfinite(opp) and top > ext + 0.5 * (opp - ext):
                    continue
                orders.append((t, 1, top, ext - 0.1 * a[t], opp, valid))
            if d == -1 and l[t - 2] - h[t] >= min_gap * a[t]:
                bot = h[t]
                if discount and np.isfinite(opp) and bot < ext - 0.5 * (ext - opp):
                    continue
                orders.append((t, -1, bot, ext + 0.1 * a[t], opp, valid))
    return orders


def combo_orders(df: pd.DataFrame, k: int = 3, k_small: int = 2, trend: str = "struct", fib=(0.5, 0.786),
                 look: int = 6, need_sweep: bool = True, need_pattern: bool = True, tp_mode: str = "struct"):
    """Setup gabungan (long; short kebalikannya):
    1. Tren searah: 'struct' = tren mayor harian & minor 4 jam naik; 'ema' = EMA20 > EMA50 (4 jam);
       'major' = hanya tren mayor harian; 'both' = struct dan ema.
    2. Kaki impuls terakhir (swing low -> swing high, fraktal k) dan harga pullback ke zona Fibonacci.
    3. Liquidity sweep: dalam `look` candle terakhir ada candle yang menembus swing low kecil (fraktal k_small)
       lalu close kembali di atasnya, tanpa menembus awal kaki impuls.
    4. Candle konfirmasi (bullish engulfing / pin bar) di candle sinyal.
    Masuk market di open berikutnya, stop di bawah low terendah sejak sweep, target puncak kaki (struct),
    ekstensi -0.272 (ext), atau -0.618 (ext618)."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = df["atr"].to_numpy()
    big = structure(df, k)
    sm = structure(df, k_small)
    shp, slp, shi, sli, minor = (big[x].to_numpy() for x in ("sh", "sl", "shi", "sli", "trend"))
    ssh, ssl = sm["sh"].to_numpy(), sm["sl"].to_numpy()
    major = df["major"].to_numpy()
    emad = np.sign(df["close"].ewm(span=20, adjust=False).mean() - df["close"].ewm(span=50, adjust=False).mean()).to_numpy()
    if id(df) not in _CC:
        _CC[id(df)] = candles(df)
    bull_c, bear_c = _CC[id(df)]
    n = len(c)
    # sweep kecil per candle
    sw_l = np.isfinite(ssl) & (l < ssl) & (c > ssl)
    sw_h = np.isfinite(ssh) & (h > ssh) & (c < ssh)
    orders = []
    for t in range(60, n - 1):
        for d in (1, -1):
            if trend in ("struct", "both") and not (major[t] == d and minor[t] == d):
                continue
            if trend == "major" and major[t] != d:
                continue
            if trend in ("ema", "both") and emad[t] != d:
                continue
            if need_pattern and not (bull_c[t] if d == 1 else bear_c[t]):
                continue
            if not (np.isfinite(shp[t]) and np.isfinite(slp[t])):
                continue
            if (d == 1 and not sli[t] < shi[t]) or (d == -1 and not shi[t] < sli[t]):
                continue
            lo, hi = slp[t], shp[t]
            leg = hi - lo
            if leg < 2 * a[t]:
                continue
            s0 = max(t - look, 0)
            if d == 1:
                zt, zb = hi - fib[0] * leg, hi - fib[1] * leg
                seg_low = l[s0:t + 1].min()
                if not (seg_low <= zt and c[t] >= zb and seg_low > lo):
                    continue
                if need_sweep and not sw_l[s0:t + 1].any():
                    continue
                stop = seg_low - 0.2 * a[t]
                target = {"struct": hi, "ext": hi + 0.272 * leg, "ext618": hi + 0.618 * leg}[tp_mode]
            else:
                zb, zt = lo + fib[0] * leg, lo + fib[1] * leg
                seg_high = h[s0:t + 1].max()
                if not (seg_high >= zb and c[t] <= zt and seg_high < hi):
                    continue
                if need_sweep and not sw_h[s0:t + 1].any():
                    continue
                stop = seg_high + 0.2 * a[t]
                target = {"struct": lo, "ext": lo - 0.272 * leg, "ext618": lo - 0.618 * leg}[tp_mode]
            orders.append((t, d, None, stop, target, 1))
    return orders
