"""Retry failed odds backfill combos."""

import subprocess

RETRY = [
    ("2026-02-21", "飯塚"),
    ("2026-02-22", "浜松"), ("2026-02-22", "飯塚"),
    ("2026-02-23", "浜松"), ("2026-02-23", "飯塚"),
    ("2026-02-24", "伊勢崎"), ("2026-02-24", "川口"),
    ("2026-02-25", "伊勢崎"), ("2026-02-25", "川口"),
    ("2026-02-26", "伊勢崎"), ("2026-02-26", "川口"),
    ("2026-02-27", "伊勢崎"),
    ("2026-02-28", "浜松"), ("2026-02-28", "飯塚"),
]

errors = 0
for i, (dt, track) in enumerate(RETRY, 1):
    print(f"[{i}/{len(RETRY)}] {dt} {track} ...", end=" ", flush=True)
    result = subprocess.run(
        ["uv", "run", "providence", "scrape", "odds", "--date", dt, "--track", track],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors += 1
        print(f"ERROR: {result.stderr.strip()[-120:]}")
    else:
        last_line = [l for l in result.stdout.strip().splitlines() if l][-1] if result.stdout.strip() else ""
        print(last_line)

print(f"\n=== RETRY COMPLETE: {len(RETRY)} combos, {errors} errors ===")
