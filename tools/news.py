"""Kumpulkan berita crypto dari RSS lalu beri skor sentimen per koin -> news.json.

Dijalankan GitHub Actions tiap 30 menit. Hanya pakai library standar Python.
Sentimen dihitung dari kata kunci di judul berita: kasar, tapi transparan.
"""
from __future__ import annotations

import email.utils
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "The Block": "https://www.theblock.co/rss.xml",
    "Bitcoinist": "https://bitcoinist.com/feed/",
}

# Kata yang dicari di judul (huruf kecil, batas kata). Ticker pendek yang ambigu tidak dipakai.
COINS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "ether", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb", "bnb chain"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["dogecoin", "doge"],
    "TAO": ["bittensor", "tao"],
    "HYPE": ["hyperliquid"],
    "ASTER": ["aster dex", "aster"],
    "XPL": ["plasma xpl", "xpl"],
    "LIT": ["lighter dex", "lighter exchange", "lighter perp", "lighter protocol", "lit token"],
    "XAU": ["gold", "xau"],
}

POSITIVE = """surge surges surged soar soars soared rally rallies rallied jump jumps jumped climb climbs climbed gain gains
rise rises rising rose breakout bullish record high highs ath all-time approve approves approved approval etf inflow inflows
adopt adopts adoption partnership partners launch launches launched upgrade upgrades buy buys buying accumulate accumulates
accumulation recover recovers recovery rebound rebounds rebounded boost boosts boosted outperform outperforms win wins
milestone integrate integrates integration greenlight backs listing listed expands expansion""".split()
NEGATIVE = """crash crashes crashed plunge plunges plunged drop drops dropped fall falls fell slump slumps slide slides tumble
tumbles tumbled sink sinks sank bearish selloff sell-off dump dumps dumped hack hacked hacks exploit exploited breach scam
fraud lawsuit sues sued charges charged ban bans banned reject rejects rejected outflow outflows liquidation liquidations
liquidated fear fears warning warns risk risks probe investigation sec delist delisted delisting low lows loss losses
collapse collapses bankrupt bankruptcy halt halts suspend suspends""".split()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (radar-crypto news bot)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def parse(source: str, data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    items = []
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        try:
            t = email.utils.parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
        except (TypeError, ValueError):
            t = None
        if title and t:
            items.append({"title": title, "link": link, "source": source, "time": t})
    return items


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


def sentiment(title: str) -> int:
    w = words(title)
    pos = sum(x in POSITIVE for x in w) + (1 if "all-time high" in title.lower() else 0)
    neg = sum(x in NEGATIVE for x in w)
    return (pos > neg) - (pos < neg)


def mentions(title: str, keys: list[str]) -> bool:
    low = " " + " ".join(words(title)) + " "
    return any(f" {k} " in low for k in keys)


def main() -> None:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=48)
    items, errors = [], {}
    for name, url in FEEDS.items():
        try:
            items += parse(name, fetch(url))
        except Exception as exc:  # satu feed gagal tidak menghentikan yang lain
            errors[name] = str(exc)[:200]
    seen, recent = set(), []
    for it in sorted(items, key=lambda x: x["time"], reverse=True):
        key = it["title"].lower()
        if it["time"] >= since and key not in seen:
            seen.add(key)
            recent.append(it)

    coins = {}
    for sym, keys in COINS.items():
        hits = [it for it in recent if mentions(it["title"], keys)]
        scored = [{"title": it["title"], "link": it["link"], "source": it["source"],
                   "time": it["time"].isoformat(), "s": sentiment(it["title"])} for it in hits]
        coins[sym] = {"pos": sum(x["s"] > 0 for x in scored), "neg": sum(x["s"] < 0 for x in scored),
                      "count": len(scored), "items": scored[:15]}

    out = {"updated": now.isoformat(), "window_hours": 48, "headlines": len(recent),
           "feeds_ok": [n for n in FEEDS if n not in errors], "feeds_failed": errors, "coins": coins}
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
