"""Backfill odds for all 2025 races, newest date first."""

import subprocess
import sys

# Read combos from DB at startup
def load_combos():
    from providence.database.engine import get_session_factory
    from sqlalchemy import text
    sf = get_session_factory()
    with sf() as s:
        rows = s.execute(text("""
            SELECT DISTINCT r.race_date, t.name
            FROM races r JOIN tracks t ON r.track_id = t.id
            WHERE r.race_date BETWEEN '2025-01-01' AND '2025-12-31'
            ORDER BY r.race_date DESC, t.name
        """)).all()
    return [(str(r[0]), r[1]) for r in rows]

COMBOS = load_combos()
print(f"Loaded {len(COMBOS)} combos to backfill", flush=True)

dates_seen = set()
done = 0
errors = 0
error_list = []
total = len(COMBOS)

for i, (dt, track) in enumerate(COMBOS, 1):
    dates_seen.add(dt)
    print(f"[{i}/{total}] {dt} {track} ...", end=" ", flush=True)
    result = subprocess.run(
        ["uv", "run", "providence", "scrape", "odds", "--date", dt, "--track", track],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors += 1
        error_list.append(f"{dt}|{track}")
        print(f"ERROR")
    else:
        last_line = [l for l in result.stdout.strip().splitlines() if l][-1] if result.stdout.strip() else ""
        print(last_line)
    done += 1

    if len(dates_seen) % 5 == 0 and (i >= total or dt != COMBOS[i][0]):
        print(f"\n=== {len(dates_seen)} days done, {done}/{total} combos, {errors} errors ===\n", flush=True)

print(f"\n=== COMPLETE: {done}/{total} combos, {len(dates_seen)} days, {errors} errors ===")
if error_list:
    print("Failed combos:")
    for e in error_list:
        print(f"  {e}")
