"""Varian strategi EMA cross: stop, cara keluar, filter. Pilih di 2022-2024, cek 2025 & 2026."""
import itertools, numpy as np, pandas as pd
import data4h, mesin as m, uji
_, btc = data4h.load("BTC")
data = {sym: m.prep(df, btc) for sym, df in (data4h.load(s) for s in data4h.WATCHLIST)}
rows = []
for (f, s), stopk, exitm, filt in itertools.product([(20, 50), (21, 55), (9, 21), (12, 26)], [1.5, 2, 3], ["cross", "trail3", "trail4", "st10_3.0", "cross+tp1"], ["none", "htf", "adx", "adx25", "adx+htf", "vol"]):
    per = []
    for sym, df in data.items():
        c, a = df["close"].to_numpy(), df["atr"].to_numpy()
        up, dn = uji.cross(df[f"ema{f}"], df[f"ema{s}"])
        sig = np.where(up, 1, np.where(dn, -1, 0))
        ff = filt.replace("adx25", "adx").replace("adx+htf", "htf")
        sg = uji.apply_filter(df, sig, ff if ff in uji.FILTERS else "none")
        if filt == "adx25": sg[df["adx"].to_numpy() < 25] = 0
        if filt == "adx+htf": sg[df["adx"].to_numpy() < 20] = 0
        if filt == "vol": sg[(df["volume"] < df["vsma"]).to_numpy()] = 0
        exs, line, t1, tr = None, None, None, None
        if exitm.startswith("cross"): exs = -sig
        if exitm == "cross+tp1": t1 = 1.5
        if exitm.startswith("trail"): tr = float(exitm[-1])
        if exitm.startswith("st"): line = df[f"stl{exitm[2:]}"].to_numpy(); line = np.where(df[f"st{exitm[2:]}"].to_numpy() == np.sign(c - line), line, np.nan)
        for t in m.simulate(df, sg, stopk * a, exs, t1, None, tr, 180, line):
            per.append({**t, "sym": sym})
    def comp(y0, y1):
        tot, dds, k = 0, [], 0
        for sym in data:
            tt = [t for t in per if t["sym"] == sym and y0 <= t["time"].year < y1]
            if not tt: continue
            eq, dd = m.compound(tt); tot += eq; dds.append(dd); k += 1
        return (tot / k / 1000 - 1) * 100, np.median(dds) * 100
    key = f"ema{f}/{s} stop{stopk} {exitm} {filt}"
    r = {"key": key}
    for lbl, y0, y1 in (("in", 2022, 2025), ("y25", 2025, 2026), ("y26", 2026, 2027)):
        tt = [t for t in per if y0 <= t["time"].year < y1]
        r[lbl] = m.summary(tt); r[lbl]["ret"], r[lbl]["dd"] = comp(y0, y1)
    rows.append(r)
pd.to_pickle(rows, "../hasil/cache4h/varian.pkl")
rows.sort(key=lambda r: -r["in"]["avg"] * np.sqrt(r["in"]["n"]))
for r in rows[:30]:
    i, a, b = r["in"], r["y25"], r["y26"]
    print(f"{r['key']:36} IN n{i['n']:4} w{i['win']:5} {i['avg']:+.3f} ret{i['ret']:+5.0f}% dd{i['dd']:3.0f} | 25 n{a['n']:4} w{a['win']:5} {a['avg']:+.3f} ret{a['ret']:+4.0f}% dd{a['dd']:3.0f} | 26 n{b['n']:4} w{b['win']:5} {b['avg']:+.3f} ret{b['ret']:+4.0f}% dd{b['dd']:3.0f}")
