"""Analisa arah pergerakan crypto.

Contoh:
    python main.py                          # BTC ETH SOL BNB XRP, timeframe harian
    python main.py BTC ETH -i 4h            # timeframe 4 jam
    python main.py data/btc.csv -s csv      # dari file CSV
"""
from __future__ import annotations

import argparse

from analyzer import chart, indicators, signals
from analyzer.data_sources import SOURCES, get_source

# Watchlist Aditya. LIT di Binance adalah koin lain, jadi dipaksa ke Bitget; XAU (emas) hanya ada di Bitget futures.
DEFAULT_SYMBOLS = ["HYPE", "ASTER", "XPL", "SOL", "BTC", "TAO", "XRP", "ETH", "DOGE", "bitget:LIT", "BNB", "XAU"]
TREND_ICON = {"BULLISH": "▲", "BEARISH": "▼", "SIDEWAYS": "►"}


def main() -> None:
    p = argparse.ArgumentParser(description="Analisa arah pergerakan harga crypto")
    p.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS, help="mis. BTC ETH SOL (default: watchlist)")
    p.add_argument("-i", "--interval", default="1d", help="15m, 1h, 4h, 1d, 1w (default 1d)")
    p.add_argument("-s", "--source", default="auto", choices=sorted(SOURCES))
    p.add_argument("-n", "--limit", type=int, default=400, help="jumlah candle yang diambil")
    p.add_argument("-o", "--out", default="charts", help="folder output grafik")
    p.add_argument("--no-chart", action="store_true", help="jangan buat grafik")
    p.add_argument("-v", "--verbose", action="store_true", help="tampilkan alasan tiap sinyal")
    args = p.parse_args()

    source = get_source(args.source)
    results = []
    for sym in args.symbols:
        label = source.normalize_symbol(sym)
        try:
            raw = source.fetch(sym, args.interval, args.limit)
            df = indicators.add_all(raw)
        except Exception as exc:  # satu koin gagal tidak menghentikan yang lain
            print(f"[!] {label}: {exc}")
            continue
        if len(df) < 60:
            print(f"[!] {label}: data terlalu sedikit ({len(df)} candle)")
            continue
        res = signals.analyze(label, df)
        results.append((res, raw.attrs.get("source", args.source)))
        if not args.no_chart:
            chart.plot(label, df, res.trend, args.out)

    if not results:
        return
    print(f"\nTimeframe: {args.interval}   (perubahan dihitung 7 candle terakhir)\n")
    print(f"{'Koin':<12}{'Harga':>14}{'Ubah':>9}{'RSI':>7}{'Skor':>6}  {'Tren':<12}Sumber")
    print("-" * 70)
    for r, src in results:
        print(f"{r.symbol:<12}{r.price:>14,.4f}{r.change_pct:>8.1f}%{r.rsi:>7.1f}{r.score:>+6d}  "
              f"{TREND_ICON[r.trend] + ' ' + r.trend:<12}{src}")
        if args.verbose:
            for note in r.notes:
                print(f"{'':14}{note}")
    if not args.no_chart:
        print(f"\nGrafik disimpan di folder: {args.out}/")
    print("\nCatatan: ini analisa teknikal otomatis, bukan saran investasi.")


if __name__ == "__main__":
    main()
