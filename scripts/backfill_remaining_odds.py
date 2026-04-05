"""Backfill remaining odds for a given year range, newest first.

Usage:
    uv run python scripts/backfill_remaining_odds.py 2025
    uv run python scripts/backfill_remaining_odds.py 2023 2026
"""

import subprocess
import sys

def load_remaining(start_year: int, end_year: int):
    from providence.database.engine import get_session_factory
    from sqlalchemy import text
    sf = get_session_factory()
    with sf() as s:
        rows = s.execute(text(f"""
            SELECT r.race_date, t.name
            FROM races r
            JOIN tracks t ON r.track_id = t.id
            WHERE r.race_date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
            AND NOT EXISTS (
                SELECT 1 FROM odds_snapshot o
                WHERE o.race_id = r.id AND o.ingestion_batch_id IS NOT NULL
            )
            GROUP BY r.race_date, t.name
            ORDER BY r.race_date DESC, t.name
        """)).all()
    return [(str(r[0]), r[1]) for r in rows]

if len(sys.argv) < 2:
    print("Usage: python scripts/backfill_remaining_odds.py <start_year> [end_year]")
    sys.exit(1)

start_year = int(sys.argv[1])
end_year = int(sys.argv[2]) if len(sys.argv) > 2 else start_year
label = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)

COMBOS = load_remaining(start_year, end_year)
print(f"[{label}] {len(COMBOS)} remaining combos to backfill", flush=True)

if not COMBOS:
    print(f"[{label}] Nothing to do.")
    sys.exit(0)

dates_seen = set()
done = 0
errors = 0
error_list = []
total = len(COMBOS)

for i, (dt, track) in enumerate(COMBOS, 1):
    dates_seen.add(dt)
    print(f"[{label}] [{i}/{total}] {dt} {track} ...", end=" ", flush=True)
    result = subprocess.run(
        ["uv", "run", "providence", "scrape", "odds", "--date", dt, "--track", track],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors += 1
        error_list.append(f"{dt}|{track}")
        print("ERROR")
    else:
        last_line = [l for l in result.stdout.strip().splitlines() if l][-1] if result.stdout.strip() else ""
        print(last_line)
    done += 1

    if len(dates_seen) % 5 == 0 and (i >= total or dt != COMBOS[i][0]):
        print(f"\n[{label}] === {len(dates_seen)} days done, {done}/{total} combos, {errors} errors ===\n", flush=True)

print(f"\n[{label}] === COMPLETE: {done}/{total} combos, {len(dates_seen)} days, {errors} errors ===")
if error_list:
    print(f"[{label}] Failed combos:")
    for e in error_list:
        print(f"  {e}")
