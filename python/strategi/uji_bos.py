"""Trend following murni struktur (tanpa EMA): masuk saat BOS/CHoCH di 4 jam, keluar saat struktur berbalik."""
import itertools, numpy as np
import uji_pa as U, pa, mesin as m
for k, maj, stopm in itertools.product([2, 3, 5], [False, True], ["swing", "2atr"]):
    per = []
    for sym, df in U.data.items():
        st = pa.structure(df, k)
        tr = st["trend"].to_numpy()
        flip = np.where((tr != np.roll(tr, 1)) & (tr != 0), tr, 0); flip[0] = 0
        sig = flip.copy()
        if maj: sig[sig != df["major"].to_numpy()] = 0
        c, a = df["close"].to_numpy(), df["atr"].to_numpy()
        if stopm == "swing":
            sd = np.where(sig == 1, c - st["sl"].to_numpy(), st["sh"].to_numpy() - c) + 0.2 * a
            sd = np.where((sd > 0.5 * a) & (sd < 6 * a), sd, 2 * a)
        else:
            sd = 2 * a
        per += [{**t, "sym": sym} for t in m.simulate(df, sig, sd, -flip, None, None, None, 400)]
    r = U.evaluate(per)
    i, a_, b = r["in"], r["y25"], r["y26"]
    print(f"BOS k{k} major{maj} stop{stopm:5} IN n{i['n']:4} w{i['win']:5} {i['avg']:+.3f} ret{i['ret']:+5.0f}% | 25 n{a_['n']:4} w{a_['win']:5} {a_['avg']:+.3f} ret{a_['ret']:+4.0f}% dd{a_['dd']:3.0f} | 26 n{b['n']:4} w{b['win']:5} {b['avg']:+.3f} ret{b['ret']:+4.0f}% dd{b['dd']:3.0f} {b['n_coin_yr']:.0f}/koin")
