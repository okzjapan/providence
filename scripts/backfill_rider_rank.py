"""Backfill rider rank (S/A/B) for all riders by re-fetching profiles.

Usage:
    uv run python scripts/backfill_rider_rank.py
"""

import asyncio
import time

from providence.config import get_settings
from providence.database.engine import get_engine, get_session_factory
from providence.database.repository import Repository
from providence.database.tables import Rider
from providence.scraper.autorace_jp import AutoraceJpScraper

from sqlalchemy import select


async def main():
    settings = get_settings()
    scraper = AutoraceJpScraper(settings)
    session_factory = get_session_factory()
    repo = Repository()

    with session_factory() as session:
        riders = session.execute(select(Rider)).scalars().all()
        total = len(riders)
        print(f"Total riders: {total}")

        null_rank = sum(1 for r in riders if r.rank is None)
        print(f"Riders without rank: {null_rank}")

        updated = 0
        errors = 0

        for i, rider in enumerate(riders):
            if rider.rank is not None:
                continue

            try:
                profile = await scraper.get_player_profile(rider.registration_number)
                if profile and profile.rank:
                    rank_str = profile.rank.value if hasattr(profile.rank, "value") else str(profile.rank)
                    rider.rank = rank_str
                    updated += 1
                    if updated % 50 == 0:
                        session.commit()
                        print(f"  [{i+1}/{total}] Updated {updated} ranks, {errors} errors")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error for {rider.registration_number}: {e}")

            if (i + 1) % 10 == 0:
                time.sleep(settings.scrape_interval_sec)

        session.commit()
        print(f"\nDone. Updated: {updated}, Errors: {errors}")

        # Verify
        with_rank = sum(1 for r in session.execute(select(Rider)).scalars().all() if r.rank is not None)
        print(f"Riders with rank: {with_rank}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
