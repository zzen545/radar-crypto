"""Uji setup gabungan tren + Fibonacci + liquidity sweep + pola candle."""
import itertools, json, numpy as np
import uji_pa as U, pa
rows = []
grid = itertools.product([3, 5], [2, 3], ["struct", "major", "ema", "both"], [(0.5, 0.786), (0.382, 0.786), (0.618, 0.886)],
                         [6, 12], [True], [True, False], ["struct", "ext", "ext618"], [1.0, 1.5], [None, 1.0])
for k, ks, tr, fib, look, sw, pat, tp, rrmin, be in grid:
    if ks >= k: continue
    per = []
    for sym, df in U.data.items():
        od = pa.combo_orders(df, k=k, k_small=ks, trend=tr, fib=fib, look=look, need_sweep=sw, need_pattern=pat, tp_mode=tp)
        per += [{**t, "sym": sym} for t in pa.simulate(df, od, rr_min=rrmin, be_at=be, rr_cap=8)]
    rows.append({"key": f"k{k}/{ks} tren={tr} fib={fib[0]}-{fib[1]} look{look} pola={pat} tp={tp} rrmin{rrmin} be{be}", **U.evaluate(per)})
json.dump(rows, open("../hasil/uji_combo.json", "w"), indent=0)
rows.sort(key=lambda r: -r["in"]["avg"] * np.sqrt(r["in"]["n"]))
for r in rows[:25]:
    i, a, b = r["in"], r["y25"], r["y26"]
    print(f"{r['key']:78} IN n{i['n']:4} w{i['win']:5} {i['avg']:+.3f} ret{i['ret']:+5.0f}% | 25 n{a['n']:4} w{a['win']:5} {a['avg']:+.3f} ret{a['ret']:+4.0f}% | 26 n{b['n']:4} w{b['win']:5} {b['avg']:+.3f} ret{b['ret']:+4.0f}% dd{b['dd']:3.0f} {b['n_coin_yr']:.0f}/koin")
