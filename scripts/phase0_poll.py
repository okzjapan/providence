"""Phase 0: Poll Hold/Today API every 2 minutes and log key fields.

Usage:
    uv run python scripts/phase0_poll.py          # run for 60 minutes
    uv run python scripts/phase0_poll.py --minutes 120
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

import httpx

BASE = "https://autorace.jp"
OUT_DIR = Path("data/phase0_spike")
INTERVAL_SEC = 120


async def fetch_today() -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(f"{BASE}/race_info/XML/Hold/Today")
        resp.raise_for_status()
        data = resp.json()
    body = data.get("body", data)
    return body.get("today", [])


FIELDS = [
    "fetch_time",
    "placeCode",
    "placeName",
    "nowRaceNo",
    "oddsRaceNo",
    "resultRaceNo",
    "telvoteTime",
    "telvoteClose",
    "raceStartTime",
    "raceName",
    "cancelFlg",
    "weather",
    "preSale",
]


async def main(minutes: int) -> None:
    out_path = OUT_DIR / f"poll_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        end_time = datetime.now().timestamp() + minutes * 60
        iteration = 0

        while datetime.now().timestamp() < end_time:
            iteration += 1
            fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                items = await fetch_today()
            except Exception as e:
                print(f"[{fetch_time}] ERROR: {e}", file=sys.stderr)
                await asyncio.sleep(INTERVAL_SEC)
                continue

            for item in items:
                row = {"fetch_time": fetch_time}
                for field in FIELDS:
                    if field != "fetch_time":
                        row[field] = item.get(field, "")
                writer.writerow(row)

            f.flush()
            summary = " | ".join(
                f"{it.get('placeName', '?')} R{it.get('nowRaceNo', '?')} tv={it.get('telvoteTime', '?')} close={it.get('telvoteClose', '?')}"
                for it in items
            )
            print(f"[{fetch_time}] #{iteration} tracks={len(items)} | {summary}")

            await asyncio.sleep(INTERVAL_SEC)

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()
    asyncio.run(main(args.minutes))
