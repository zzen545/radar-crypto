# Analisa Pergerakan Crypto

Program Python yang mengambil data harga (OHLCV) dari API publik Binance, dan otomatis dari Bitget untuk koin yang tidak ada (atau datanya kurang) di Binance, menghitung
indikator teknikal, lalu memberi ringkasan arah tren per koin: **BULLISH**, **BEARISH**,
atau **SIDEWAYS**, lengkap dengan grafik PNG.

## Dashboard di browser (tanpa Python)

Buka `dashboard.html` dengan dobel-klik (Chrome, Edge, atau Firefox). Data diambil langsung
dari Binance dan Bitget, dianalisa dengan aturan yang sama seperti program Python, lalu
diperbarui otomatis tiap 5 menit. Ganti daftar koin atau timeframe di bagian atas, lalu klik
**Analisa**. Klik baris koin untuk melihat grafik harga, MA, Bollinger, RSI, MACD, dan alasan sinyalnya.

## Cara pakai (Python)

```bash
pip install -r requirements.txt

python main.py                    # watchlist (HYPE ASTER XPL SOL BTC TAO XRP ETH DOGE LIT BNB XAU), harian
python main.py BTC ETH DOGE -v    # koin pilihan + alasan tiap sinyal
python main.py BTC -i 4h          # timeframe 4 jam (15m, 1h, 4h, 1d, 1w)
python main.py bitget:LIT         # paksa ambil dari Bitget (ticker sama tapi koin beda di Binance)
python main.py BTC -s bitget      # hanya Bitget (spot, lalu futures USDT-M mis. XAUUSDT)
python main.py data/btc.csv -s csv  # dari file CSV (time/date, open, high, low, close, volume)
```

Grafik disimpan di folder `charts/`.

## Cara skor dihitung

| Sinyal | Poin |
|---|---|
| Harga di atas / bawah MA20, MA50, MA200 | +1 / −1 masing-masing |
| MA50 di atas / bawah MA200 | +1 / −1 |
| MACD di atas / bawah garis signal | +1 / −1 |
| RSI > 55 / < 45 (antara 30–70) | +1 / −1 |
| Volume > 1,5× rata-rata 20 candle | +1 jika harga naik, −1 jika turun |

Skor ≥ 3 → BULLISH, ≤ −3 → BEARISH, selain itu SIDEWAYS. RSI overbought/oversold,
tembusan Bollinger Bands, dan Bollinger yang menyempit ditampilkan sebagai peringatan
(dengan `-v`) tanpa mengubah skor.

## Struktur

- `analyzer/data_sources.py`: sumber data (auto = Binance lalu Bitget, Binance, Bitget spot + futures, CSV). Tambah sumber baru, misalnya
  saham IDX, dengan membuat turunan `DataSource` dan mendaftarkannya di `SOURCES`.
- `analyzer/indicators.py`: MA, EMA, RSI, MACD, Bollinger Bands, volume.
- `analyzer/signals.py`: penggabungan sinyal jadi skor dan tren.
- `analyzer/chart.py`: grafik harga, volume, RSI, MACD.
- `tests/`: uji dengan data buatan (`python -m pytest tests`).

> Ini analisa teknikal otomatis, bukan saran investasi.
