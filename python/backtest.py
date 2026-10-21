"""Backtest aturan dashboard Radar Tren Crypto di data harian historis.

Yang disimulasikan (sama dengan dashboard):
- Skor = teknikal (analyzer/signals.py) + volume (OBV, porsi pembeli agresif).
  Transaksi real-time dan berita tidak bisa diuji karena tidak ada data historisnya.
- BULLISH (skor >= 3) -> LONG, BEARISH (skor <= -3) -> SHORT, selain itu tidak masuk.
- Entry: limit order di harga close -/+ 0,25 ATR, berlaku 3 candle. Tidak terisi = batal.
- Stop loss di balik support/resistance terdekat (+0,5 ATR), atau 1,5 ATR bila jaraknya tidak wajar.
- Setengah posisi ditutup di TP1 (1,5R), lalu SL digeser ke entry; sisanya di TP2 (3R).
- Bila SL dan TP tersentuh di candle yang sama, dianggap kena SL dulu (konservatif).
- Maksimal hold 30 candle, lalu ditutup di harga close. Biaya 0,1% per transaksi.

Contoh:
    python backtest.py                 # watchlist default
    python backtest.py BTC ETH --json hasil.json
"""
from __future__ import annotations

import argparse
import functools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

from analyzer import indicators, signals
from analyzer.data_sources import BitgetSource

WATCHLIST = ["HYPE", "ASTER", "XPL", "SOL", "BTC", "TAO", "XRP", "ETH", "DOGE", "bitget:LIT", "BNB", "XAU"]
FEE = 0.001
MAX_HOLD = 30
FILL_DAYS = 3
WARMUP = 210


# ---------- data ----------
def _klines(params: dict) -> list:
    for base in ("https://api.binance.com", "https://data-api.binance.vision"):
        try:
            r = requests.get(f"{base}/api/v3/klines", params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            continue
    raise ConnectionError("Binance tidak merespons")


def fetch_binance(sym: str, pages: int = 3) -> pd.DataFrame:
    """Sampai 3000 candle harian (±8 tahun), diambil mundur 1000 per request."""
    rows: list = []
    end = None
    for _ in range(pages):
        params = {"symbol": sym, "interval": "1d", "limit": 1000, **({"endTime": end} if end else {})}
        page = _klines(params)
        if not page:
            break
        rows = page + rows
        if len(page) < 1000:
            break
        end = page[0][0] - 1
    now_ms = pd.Timestamp.now(tz="UTC").value // 1_000_000
    rows = [k for k in rows if k[6] < now_ms]
    df = pd.DataFrame({"time": [k[0] for k in rows], "open": [k[1] for k in rows], "high": [k[2] for k in rows],
                       "low": [k[3] for k in rows], "close": [k[4] for k in rows], "volume": [k[5] for k in rows],
                       "taker_buy": [k[9] for k in rows]})
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time").astype(float)


@functools.lru_cache(maxsize=None)
def load(symbol: str) -> tuple[str, str, pd.DataFrame]:
    forced, _, rest = symbol.rpartition(":")
    sym = BitgetSource().normalize_symbol(rest)
    if forced != "bitget":
        try:
            df = fetch_binance(sym)
            if len(df) >= 260:
                return sym, "binance", df
        except Exception:
            pass
    df = BitgetSource().fetch(rest, "1d", 1000)
    df["taker_buy"] = np.nan
    return sym, "bitget", df


# ---------- sinyal ----------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def volume_score(df: pd.DataFrame, obv: pd.Series, obv_sma: pd.Series, t: int) -> int:
    score = 0
    if obv.iloc[t] > obv_sma.iloc[t] and obv.iloc[t] > obv.iloc[t - 10]:
        score += 1
    elif obv.iloc[t] < obv_sma.iloc[t] and obv.iloc[t] < obv.iloc[t - 10]:
        score -= 1
    w7 = df.iloc[t - 6:t + 1]
    if w7["taker_buy"].notna().all():
        share = w7["taker_buy"].sum() / w7["volume"].sum() * 100
        score += 1 if share > 55 else -1 if share < 45 else 0
    else:
        w20 = df.iloc[t - 19:t + 1]
        share = w20.loc[w20["close"] >= w20["open"], "volume"].sum() / w20["volume"].sum() * 100
        score += 1 if share > 60 else -1 if share < 40 else 0
    return score


def levels(df: pd.DataFrame, t: int, look: int = 150, k: int = 3) -> tuple[float, float]:
    hi, lo, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].iloc[t]
    n = t + 1
    sup, res = [], []
    for i in range(max(k, n - look), n - k):
        if hi[i] >= hi[i - k:i + k + 1].max():
            res.append(hi[i])
        if lo[i] <= lo[i - k:i + k + 1].min():
            sup.append(lo[i])
    below, above = [x for x in sup if x < close], [x for x in res if x > close]
    support = max(below) if below else lo[max(0, n - 60):n].min()
    resistance = min(above) if above else hi[max(0, n - 60):n].max()
    return support, resistance


def make_plan(side: str, close: float, a: float, support: float, resistance: float) -> dict:
    d = 1 if side == "LONG" else -1
    entry = close - d * 0.25 * a
    guard = support if side == "LONG" else resistance
    stop = guard - d * 0.5 * a
    if not (0.75 * a <= d * (entry - stop) <= 3 * a):
        stop = entry - d * 1.5 * a
    risk = d * (entry - stop)
    return {"side": side, "entry": entry, "stop": stop, "risk": risk,
            "tp1": entry + d * 1.5 * risk, "tp2": entry + d * 3 * risk}


# ---------- simulasi ----------
@dataclass
class Trade:
    symbol: str
    side: str
    signal_date: str
    entry_date: str
    exit_date: str
    days: int
    score: int
    r: float
    tp1: bool
    outcome: str


def simulate(df: pd.DataFrame, plan: dict, start: int) -> tuple[int, int, float, bool, str] | None:
    """Kembalikan (indeks entry, indeks exit, R bersih, TP1 kena?, hasil) atau None bila tidak terisi."""
    d = 1 if plan["side"] == "LONG" else -1
    hi, lo, cl = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    entry, risk = plan["entry"], plan["risk"]
    fee_r = FEE * 2 * entry / risk
    fill = next((i for i in range(start, min(start + FILL_DAYS, len(df)))
                 if (lo[i] <= entry if d == 1 else hi[i] >= entry)), None)
    if fill is None:
        return None
    stop, half_done, realized = plan["stop"], False, 0.0
    for i in range(fill, min(fill + MAX_HOLD, len(df))):
        adverse = lo[i] if d == 1 else hi[i]
        favor = hi[i] if d == 1 else lo[i]
        if d * (adverse - stop) <= 0:  # kena stop (SL awal atau SL di entry)
            rest = 0.5 if half_done else 1.0
            r = realized + rest * d * (stop - entry) / risk
            return fill, i, r - fee_r, half_done, "TP1 lalu BE" if half_done else "SL"
        if not half_done and d * (favor - plan["tp1"]) >= 0:
            half_done, realized, stop = True, 0.5 * 1.5, entry
        if half_done and d * (favor - plan["tp2"]) >= 0:
            return fill, i, realized + 0.5 * 3 - fee_r, True, "TP2"
        if i == len(df) - 1:
            return None  # masih terbuka di akhir data: tidak dihitung
    last = min(fill + MAX_HOLD, len(df)) - 1
    rest = 0.5 if half_done else 1.0
    return fill, last, realized + rest * d * (cl[last] - entry) / risk - fee_r, half_done, "Waktu habis"


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, down = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def weekly_trend(df: pd.DataFrame) -> pd.Series:
    """+1 bila close mingguan terakhir yang SUDAH selesai di atas EMA20 mingguan, -1 bila di bawah."""
    w = df["close"].resample("W-SUN").last()
    trend = np.sign(w - w.ewm(span=20, adjust=False).mean())
    # pakai minggu sebelumnya agar tidak melihat masa depan
    return trend.shift(1).reindex(df.index, method="ffill").fillna(0)


BTC_MA = 50


def btc_context(df: pd.DataFrame) -> pd.DataFrame:
    """Data BTC yang sudah diketahui saat candle koin ini close (tanpa melihat masa depan)."""
    _, _, btc = load("BTC")
    b = pd.DataFrame({"btc": btc["close"], "btc_sma50": btc["close"].rolling(BTC_MA).mean()})
    b.index = b.index + pd.Timedelta(days=1)  # waktu close candle BTC
    out = b.reindex(df.index + pd.Timedelta(days=1), method="ffill")
    out.index = df.index
    return out


def rs_signal(df: pd.DataFrame) -> pd.Series:
    """+1 bila koin lebih kuat dari BTC (rasio di atas SMA50 dan naik 20 hari), -1 bila lebih lemah."""
    ratio = df["close"] / df["btc"]
    sma = ratio.rolling(50).mean()
    up = (ratio > sma) & (ratio > ratio.shift(20))
    down = (ratio < sma) & (ratio < ratio.shift(20))
    return pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=df.index)


def backtest(symbol: str, use_volume: bool = True, min_adx: float = 0, mtf: bool = False, rs: str = "",
             btc_regime: bool = False, short_bear: bool = False, no_short: bool = False) -> tuple[list[Trade], dict]:
    sym, source, raw = load(symbol)
    df = indicators.add_all(raw)
    df = df.join(btc_context(df))
    df["rs"] = rs_signal(df) if sym != "BTCUSDT" else 0
    df["atr"] = atr(df)
    df["adx"] = adx(df)
    df["wtrend"] = weekly_trend(df)
    sign = np.sign(df["close"].diff()).fillna(0)
    obv = (sign * df["volume"]).cumsum()
    obv_sma = obv.rolling(20).mean()
    trades: list[Trade] = []
    t = WARMUP
    while t < len(df) - 1:
        res = signals.analyze(sym, df.iloc[:t + 1])
        score = res.score + (volume_score(df, obv, obv_sma, t) if use_volume else 0)
        if rs == "score":
            score += int(df["rs"].iloc[t])
        side = "LONG" if score >= 3 else "SHORT" if score <= -3 else None
        d = 1 if side == "LONG" else -1
        if side and rs == "filter" and sym != "BTCUSDT" and df["rs"].iloc[t] == -d:
            side = None
        if side and btc_regime and np.sign(df["btc"].iloc[t] - df["btc_sma50"].iloc[t]) == -d:
            side = None
        if side == "SHORT" and (no_short or (short_bear and df["close"].iloc[t] > df["sma200"].iloc[t])):
            side = None
        if side and min_adx and df["adx"].iloc[t] < min_adx:
            side = None
        if side and mtf and df["wtrend"].iloc[t] != (1 if side == "LONG" else -1):
            side = None
        if side is None:
            t += 1
            continue
        support, resistance = levels(df, t)
        plan = make_plan(side, df["close"].iloc[t], df["atr"].iloc[t], support, resistance)
        out = simulate(df, plan, t + 1)
        if out is None:
            t += 1
            continue
        fill, exit_i, r, tp1, outcome = out
        trades.append(Trade(sym, side, str(df.index[t].date()), str(df.index[fill].date()), str(df.index[exit_i].date()),
                            exit_i - fill + 1, score, round(r, 3), tp1, outcome))
        t = exit_i + 1  # satu posisi per koin
    info = {"symbol": sym, "source": source, "start": str(df.index[WARMUP].date()), "end": str(df.index[-1].date()),
            "days": len(df) - WARMUP, "hold_pct": round((df["close"].iloc[-1] / df["close"].iloc[WARMUP] - 1) * 100, 1)}
    return trades, info


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    r = np.array([t.r for t in trades])
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum).max())
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": len(trades),
        "win_rate": round(float((r > 0).mean() * 100), 1),
        "tp1_rate": round(float(np.mean([t.tp1 for t in trades]) * 100), 1),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 2),
        "profit_factor": round(float(gains / losses), 2) if losses else None,
        "max_dd_r": round(dd, 2),
        "avg_days": round(float(np.mean([t.days for t in trades])), 1),
        "median_days_tp1_wins": float(np.median([t.days for t in trades if t.r > 0])) if (r > 0).any() else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest aturan Radar Tren Crypto")
    p.add_argument("symbols", nargs="*", default=WATCHLIST)
    p.add_argument("--no-volume", action="store_true", help="skor teknikal saja")
    p.add_argument("--adx", type=float, default=0, help="hanya masuk bila ADX >= nilai ini (mis. 20)")
    p.add_argument("--mtf", action="store_true", help="hanya masuk searah tren mingguan")
    p.add_argument("--rs", choices=["filter", "score"], default="", help="kekuatan relatif vs BTC: filter atau tambahan skor")
    p.add_argument("--btc-regime", action="store_true", help="long hanya saat BTC di atas MA50, short hanya saat di bawah")
    p.add_argument("--btc-ma", type=int, default=50, help="panjang MA BTC untuk --btc-regime")
    p.add_argument("--short-bear", action="store_true", help="short hanya bila harga di bawah MA200")
    p.add_argument("--no-short", action="store_true", help="hanya LONG")
    p.add_argument("--json", help="simpan hasil lengkap ke file JSON")
    args = p.parse_args()
    global BTC_MA
    BTC_MA = args.btc_ma

    all_trades, per_coin = [], []
    for s in args.symbols:
        try:
            trades, info = backtest(s, use_volume=not args.no_volume, min_adx=args.adx, mtf=args.mtf, rs=args.rs,
                                   btc_regime=args.btc_regime, short_bear=args.short_bear, no_short=args.no_short)
        except Exception as exc:
            print(f"[!] {s}: {exc}")
            continue
        st = stats(trades)
        per_coin.append({**info, **st})
        all_trades += trades
        print(f"{info['symbol']:<10} {info['source']:<8} {info['start']}→{info['end']}  trade {st['trades']:>3}  "
              f"win {st.get('win_rate', 0):>5}%  total {st.get('total_r', 0):>+7}R  hold {st.get('avg_days', 0):>4} hari  "
              f"(buy&hold {info['hold_pct']:+.0f}%)")
    total = stats(all_trades)
    by_side = {side: stats([t for t in all_trades if t.side == side]) for side in ("LONG", "SHORT")}
    print("\nTOTAL", json.dumps(total))
    print("LONG ", json.dumps(by_side["LONG"]))
    print("SHORT", json.dumps(by_side["SHORT"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rules": {"fee": FEE, "max_hold": MAX_HOLD, "fill_days": FILL_DAYS, "volume": not args.no_volume,
                                 "adx": args.adx, "mtf": args.mtf},
                       "total": total, "by_side": by_side, "per_coin": per_coin,
                       "trades": [asdict(t) for t in all_trades]}, f, indent=1)


if __name__ == "__main__":
    main()
