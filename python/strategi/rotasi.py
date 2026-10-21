"""Rotasi momentum antar koin watchlist: tiap rebalance, long koin terkuat dan short terlemah.
Leverage 1x total (nilai posisi = modal), biaya 0,08% per sisi dari perputaran posisi."""
import itertools, numpy as np, pandas as pd
import data4h, mesin as m

closes = {}
for s in data4h.WATCHLIST:
    sym, df = data4h.load(s)
    closes[sym[:-4]] = df["close"].resample("1D").last()
px = pd.DataFrame(closes).sort_index()
ret = px.pct_change()
btc_up = (px["BTC"] > px["BTC"].rolling(50).mean())
own_up = px > px.rolling(50, min_periods=20).mean()

def run(look, reb, k, mode, volw):
    mom = px / px.shift(look) - 1
    vol = ret.rolling(30).std()
    score = mom / vol if volw else mom
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)
    for i, t in enumerate(px.index):
        if i % reb == 0:
            s = score.loc[t].dropna()
            s = s[px.loc[:t].notna().sum() >= 60]  # minimal 60 hari data
            new = pd.Series(0.0, index=px.columns)
            if len(s) >= 2 * k:
                top, bot = s.nlargest(k).index, s.nsmallest(k).index
                if mode == "ls":
                    new[top], new[bot] = 0.5 / k, -0.5 / k
                elif mode == "long":
                    new[top] = 1 / k
                elif mode == "long|btc":
                    if btc_up.loc[t]: new[top] = 1 / k
                elif mode == "long|own":
                    ok = [c for c in top if own_up.loc[t, c]]
                    new[ok] = 1 / k
                elif mode == "long|both":
                    ok = [c for c in top if own_up.loc[t, c]] if btc_up.loc[t] else []
                    new[ok] = 1 / k
                elif mode == "long+btc":
                    if btc_up.loc[t]: new[top] = 1 / k
                    else: new[bot] = -1 / k
            cur = new
        w.loc[t] = cur
    wpos = w.shift(1).fillna(0)  # bobot ditentukan di close, berlaku besoknya
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1).shift(1).fillna(0)
    pnl = (wpos * ret.fillna(0)).sum(axis=1) - turn * m.FEE
    return pnl

rows = []
for look, reb, k, mode, volw in itertools.product([7, 14, 30], [3, 7], [1, 2, 3], ["long", "long|btc", "long|own", "long|both"], [False, True]):
    pnl = run(look, reb, k, mode, volw)
    r = {"key": f"look{look} reb{reb} k{k} {mode} {'volw' if volw else ''}"}
    for y in range(2022, 2027):
        p = pnl[pnl.index.year == y]
        eq = (1 + p).cumprod()
        r[y] = (eq.iloc[-1] - 1) * 100
        r[f"dd{y}"] = (1 - eq / eq.cummax()).max() * 100
    r["is"] = np.mean([r[y] for y in (2022, 2023, 2024)])
    rows.append(r)
import json; json.dump(rows, open('../hasil/rotasi.json','w'), indent=0)
rows.sort(key=lambda r: -r["is"])
for r in rows[:30]:
    print(f"{r['key']:32} " + " ".join(f"{y}:{r[y]:+5.0f}%(dd{r[f'dd{y}']:.0f})" for y in range(2022, 2027)))
