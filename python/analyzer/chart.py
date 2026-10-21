"""Grafik harga + indikator ke file PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

UP, DOWN = "#1a9e5c", "#d1453b"


def plot(symbol: str, df: pd.DataFrame, trend: str, out_dir: str | Path, bars: int = 150) -> Path:
    df = df.tail(bars)
    fig, (ax_p, ax_v, ax_r, ax_m) = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1.3, 1.3]}
    )
    color = {"BULLISH": UP, "BEARISH": DOWN}.get(trend, "#888888")
    fig.suptitle(f"{symbol}  ·  {trend}", fontsize=14, fontweight="bold", color=color)

    ax_p.plot(df.index, df["close"], color="#222222", lw=1.4, label="Close")
    ax_p.plot(df.index, df["sma20"], lw=1, label="MA20")
    ax_p.plot(df.index, df["sma50"], lw=1, label="MA50")
    if df["sma200"].notna().any():
        ax_p.plot(df.index, df["sma200"], lw=1, label="MA200")
    ax_p.fill_between(df.index, df["bb_lower"], df["bb_upper"], color="#7aa6d8", alpha=0.15, label="Bollinger")
    ax_p.legend(loc="upper left", fontsize=8, ncol=5)
    ax_p.grid(alpha=0.3)

    vol_colors = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    ax_v.bar(df.index, df["volume"], color=vol_colors, width=0.8 * _bar_width(df))
    ax_v.plot(df.index, df["vol_sma20"], color="#444444", lw=1)
    ax_v.set_ylabel("Volume", fontsize=8)

    ax_r.plot(df.index, df["rsi"], color="#7b4fb8", lw=1)
    ax_r.axhline(70, color=DOWN, ls="--", lw=0.8)
    ax_r.axhline(30, color=UP, ls="--", lw=0.8)
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("RSI", fontsize=8)

    ax_m.plot(df.index, df["macd"], lw=1, label="MACD")
    ax_m.plot(df.index, df["macd_signal"], lw=1, label="Signal")
    ax_m.bar(df.index, df["macd_hist"], color=[UP if h >= 0 else DOWN for h in df["macd_hist"]],
             width=0.8 * _bar_width(df))
    ax_m.legend(loc="upper left", fontsize=8)
    ax_m.set_ylabel("MACD", fontsize=8)

    for ax in (ax_v, ax_r, ax_m):
        ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(symbol).stem.replace('/', '_')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _bar_width(df: pd.DataFrame) -> float:
    """Lebar bar dalam satuan hari (matplotlib date units)."""
    if len(df) < 2:
        return 1.0
    return (df.index[-1] - df.index[-2]).total_seconds() / 86400
