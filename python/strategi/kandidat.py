import sys, numpy as np, pandas as pd
import data4h, mesin as m, uji
CANDS = sys.argv[1:] or ["ema 20/50 | own | none", "ema 20/50 | own | adx", "ema 20/50 | own | htf", "macd cross | own | btc+htf", "ema 50/200 | own | none", "ema 21/55 | own | htf"]
_, btc = data4h.load("BTC")
data = {sym: m.prep(df, btc) for sym, df in (data4h.load(s) for s in data4h.WATCHLIST)}
for cand in CANDS:
    name, xn, f = [x.strip() for x in cand.split("|")]
    per = {}
    for sym, df in data.items():
        for nm, sig, sd, exs, line, kind in uji.strategies(df):
            if nm != name or (kind == "own") != (xn == "own"):
                continue
            t1, t2, tr, mh = (None, None, None, 180) if xn == "own" else uji.EXITS_TP[xn]
            per[sym] = m.simulate(df, uji.apply_filter(df, sig, f), sd, exs, t1, t2, tr, mh, line)
    print("==", cand)
    for per_label, lo, hi in (("IN 22-24", "2022", "2025"), ("2025", "2025", "2026"), ("2026", "2026", "2027")):
        allt = [t for v in per.values() for t in v if lo <= str(t["time"].year) < hi]
        s = m.summary(allt)
        tot, dds = 0, []
        coins = [k for k, v in per.items() if any(lo <= str(t["time"].year) < hi for t in v)]
        for sym in coins:
            eq, dd = m.compound([t for t in per[sym] if lo <= str(t["time"].year) < hi]); tot += eq; dds.append(dd)
        lev = np.median([1 / t["risk_pct"] * 0.05 for t in allt]) if allt else 0
        print(f"  {per_label:8} n{s['n']:4} win {s['win']}% avg {s['avg']:+.3f} pf {s['pf']}  | $1000x{len(coins)} -> {tot:,.0f} ({(tot/len(coins)/1000-1)*100:+.0f}%) maxDD med {np.median(dds)*100:.0f}% max {max(dds)*100:.0f}% lev~{lev:.1f}x")
