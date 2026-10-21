"""Menggabungkan indikator menjadi ringkasan tren: BULLISH / BEARISH / SIDEWAYS."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Analysis:
    symbol: str
    price: float
    change_pct: float
    score: int
    trend: str
    rsi: float
    notes: list[str] = field(default_factory=list)


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else 0.0


def _fmt(x: float) -> str:
    """Format harga tanpa notasi ilmiah: 83,210 / 2,521 / 110.5 / 1.389 / 0.0001234."""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.4g}"
    return f"{x:.4g}" if abs(x) >= 1e-4 else f"{x:.8f}".rstrip("0")


def analyze(symbol: str, df: pd.DataFrame, lookback: int = 7) -> Analysis:
    """Beri skor dari beberapa sinyal. Skor >= 3 bullish, <= -3 bearish, sisanya sideways."""
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    notes: list[str] = []

    def vote(points: int, note: str) -> None:
        nonlocal score
        score += points
        notes.append(f"{'+' if points > 0 else '-' if points < 0 else '·'} {note}")

    # 1. Posisi harga terhadap moving average
    for col, label in (("sma20", "MA20"), ("sma50", "MA50"), ("sma200", "MA200")):
        if pd.notna(last[col]):
            if last["close"] > last[col]:
                vote(1, f"Harga di atas {label} ({_fmt(last[col])})")
            else:
                vote(-1, f"Harga di bawah {label} ({_fmt(last[col])})")

    # 2. Susunan MA (golden/death cross jangka menengah)
    if pd.notna(last["sma50"]) and pd.notna(last["sma200"]):
        if last["sma50"] > last["sma200"]:
            vote(1, "MA50 di atas MA200 (tren besar naik)")
        else:
            vote(-1, "MA50 di bawah MA200 (tren besar turun)")

    # 3. MACD
    if last["macd"] > last["macd_signal"]:
        crossed = prev["macd"] <= prev["macd_signal"]
        vote(1, "MACD di atas signal" + (" (baru cross naik)" if crossed else ""))
    else:
        crossed = prev["macd"] >= prev["macd_signal"]
        vote(-1, "MACD di bawah signal" + (" (baru cross turun)" if crossed else ""))

    # 4. RSI: momentum, plus peringatan jenuh beli/jual
    r = last["rsi"]
    if r >= 70:
        vote(0, f"RSI {r:.1f}: overbought, rawan koreksi")
    elif r <= 30:
        vote(0, f"RSI {r:.1f}: oversold, rawan pantulan")
    elif r > 55:
        vote(1, f"RSI {r:.1f}: momentum positif")
    elif r < 45:
        vote(-1, f"RSI {r:.1f}: momentum negatif")
    else:
        vote(0, f"RSI {r:.1f}: netral")

    # 5. Bollinger Bands
    if pd.notna(last["bb_upper"]):
        width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"] * 100
        if last["close"] > last["bb_upper"]:
            vote(0, "Harga menembus Bollinger atas (kuat tapi overextended)")
        elif last["close"] < last["bb_lower"]:
            vote(0, "Harga menembus Bollinger bawah (lemah tapi overextended)")
        if width < 5:
            vote(0, f"Bollinger menyempit ({width:.1f}%): potensi pergerakan besar")

    # 6. Volume mengkonfirmasi arah?
    if pd.notna(last["vol_sma20"]) and last["volume"] > 1.5 * last["vol_sma20"]:
        up = last["close"] > prev["close"]
        vote(1 if up else -1, f"Volume tinggi ({last['volume'] / last['vol_sma20']:.1f}x rata-rata) saat harga {'naik' if up else 'turun'}")

    trend = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "SIDEWAYS"
    ref = df["close"].iloc[-1 - lookback] if len(df) > lookback else df["close"].iloc[0]
    return Analysis(
        symbol=symbol,
        price=float(last["close"]),
        change_pct=_pct(last["close"], ref),
        score=score,
        trend=trend,
        rsi=float(r),
        notes=notes,
    )
