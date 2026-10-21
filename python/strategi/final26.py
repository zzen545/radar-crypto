import numpy as np, pandas as pd, json
import data4h, mesin as m, uji
_, btc = data4h.load("BTC")
res = {}
for s in data4h.WATCHLIST:
    sym, df = data4h.load(s); df = m.prep(df, btc)
    up, dn = uji.cross(df["ema20"], df["ema50"]); sig = np.where(up, 1, np.where(dn, -1, 0))
    tr = m.simulate(df, sig, 2 * df["atr"].to_numpy(), -sig, None, None, None, 180)
    res[sym] = tr
for yr in (2025, 2026):
    print("==", yr)
    tot = {0.02: 0, 0.05: 0}; n = w = 0
    for sym, tr in res.items():
        tt = [t for t in tr if t["time"].year == yr]
        if not tt: continue
        e5, d5 = m.compound(tt, 0.05); e2, d2 = m.compound(tt, 0.02)
        tot[0.05] += e5; tot[0.02] += e2
        ww = sum(t["r"] > 0 for t in tt); n += len(tt); w += ww
        best = max(t["r"] for t in tt)
        lng = sum(t["side"] == 1 for t in tt)
        print(f"{sym[:-4]:6} n{len(tt):3} (L{lng}) win {ww:2} | 5%: {e5:6.0f} dd {d5*100:3.0f}% | 2%: {e2:6.0f} dd {d2*100:3.0f}% | best {best:+.1f}R")
    print("TOTAL", n, w, {k: round(v) for k, v in tot.items()})
for yr in (2022, 2023, 2024):
    tot=k=0; dds=[]
    for sym,tr in res.items():
        tt=[t for t in tr if t["time"].year==yr]
        if not tt: continue
        e,d=m.compound(tt,0.02); tot+=e; k+=1; dds.append(d)
    print(yr, k, round((tot/k/1000-1)*100), "dd med", round(np.median(dds)*100))
