import itertools, json, numpy as np
import uji_pa as U, pa
rows = []
for k, look, maj, mino, disc, gap, rrmin, be in itertools.product([3, 5], [6, 12, 24], [True, False], [False, True], [True, False], [0.2, 0.4], [1.0, 2.0], [None, 1.0]):
    if not maj and not mino: continue
    per = []
    for sym, df in U.data.items():
        od = pa.smc_orders(df, k=k, sweep_look=look, need_major=maj, need_minor=mino, discount=disc, min_gap=gap)
        per += [{**t, "sym": sym} for t in pa.simulate(df, od, rr_min=rrmin, be_at=be)]
    rows.append({"key": f"smc k{k} look{look} major{maj} minor{mino} disc{disc} gap{gap} rrmin{rrmin} be{be}", **U.evaluate(per)})
json.dump(rows, open("../hasil/uji_smc.json", "w"), indent=0)
rows.sort(key=lambda r: -r["in"]["avg"] * np.sqrt(r["in"]["n"]))
for r in rows[:20]:
    i, a, b = r["in"], r["y25"], r["y26"]
    print(f"{r['key']:75} IN n{i['n']:4} w{i['win']:5} {i['avg']:+.3f} ret{i['ret']:+5.0f}% | 25 n{a['n']:4} w{a['win']:5} {a['avg']:+.3f} ret{a['ret']:+4.0f}% | 26 n{b['n']:4} w{b['win']:5} {b['avg']:+.3f} ret{b['ret']:+4.0f}% dd{b['dd']:3.0f}")
