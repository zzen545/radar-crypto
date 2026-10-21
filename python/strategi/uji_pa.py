"""Grid strategi price action. Pilih di 2022-2024, cek 2025 & 2026."""
import itertools, json, numpy as np, pandas as pd
import data4h, mesin as m, pa

data = {}
for s in data4h.WATCHLIST:
    sym, df = data4h.load(s)
    df = df.copy(); df["atr"] = m.atr(df); df["major"] = pa.major_trend(df)
    data[sym] = df

def evaluate(per):
    r = {}
    for lbl, y0, y1 in (("in", 2022, 2025), ("y25", 2025, 2026), ("y26", 2026, 2027)):
        tt = [t for t in per if y0 <= t["time"].year < y1]
        r[lbl] = m.summary(tt)
        tot = k = 0; dds = []
        for sym in data:
            ts = [t for t in tt if t["sym"] == sym]
            if not ts: continue
            e, dd = m.compound(ts, 0.05); tot += e; k += 1; dds.append(dd)
        r[lbl]["ret"] = (tot / k / 1000 - 1) * 100 if k else 0
        r[lbl]["dd"] = float(np.median(dds) * 100) if dds else 0
        r[lbl]["n_coin_yr"] = r[lbl]["n"] / max(k, 1) / (y1 - y0)
    return r

def main():
    configs = []
    for k, kind, maj, mino, pat, tpm, rrmin, be in itertools.product(
            [3, 5], ["fib", "sweep"], [True, False], [False, True], [True, False], ["struct", "ext"], [1.0, 1.5], [None, 1.0]):
        if kind == "sweep" and tpm == "ext": continue
        if not maj and not mino: continue
        configs.append(("setup", dict(k=k, kind=kind, need_major=maj, need_minor=mino, need_pattern=pat, tp_mode=tpm), dict(rr_min=rrmin, be_at=be)))
    for k, maj, mino, gap, rrmin, be in itertools.product([3, 5], [True, False], [False, True], [0.3, 0.6], [1.0, 1.5], [None, 1.0]):
        if not maj and not mino: continue
        configs.append(("fvg", dict(k=k, need_major=maj, need_minor=mino, min_gap=gap), dict(rr_min=rrmin, be_at=be)))

    cache = {}
    rows = []
    for kind, sa, sim in configs:
        key_o = (kind, tuple(sorted(sa.items())))
        per = []
        for sym, df in data.items():
            ck = (sym,) + key_o
            if ck not in cache:
                cache[ck] = pa.setups(df, **sa) if kind == "setup" else pa.fvg_orders(df, **sa)
            per += [{**t, "sym": sym} for t in pa.simulate(df, cache[ck], **sim)]
        name = f"{kind} " + " ".join(f"{k}={v}" for k, v in sa.items()) + " | " + " ".join(f"{k}={v}" for k, v in sim.items())
        rows.append({"key": name, **evaluate(per)})
    json.dump(rows, open("../hasil/uji_pa.json", "w"), indent=0)
    rows.sort(key=lambda r: -r["in"]["avg"] * np.sqrt(r["in"]["n"]))
    for r in rows[:35]:
        i, a, b = r["in"], r["y25"], r["y26"]
        print(f"{r['key'][:95]:95} IN n{i['n']:4} w{i['win']:5} {i['avg']:+.3f} | 25 n{a['n']:4} w{a['win']:5} {a['avg']:+.3f} ret{a['ret']:+4.0f}% | 26 n{b['n']:4} w{b['win']:5} {b['avg']:+.3f} ret{b['ret']:+4.0f}% dd{b['dd']:3.0f} {b['n_coin_yr']:.0f}/koin")


if __name__ == "__main__":
    main()
