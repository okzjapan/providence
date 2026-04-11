#!/usr/bin/env python3
"""Re-parse local JRDB files to populate new DB columns.

Reads already-extracted files from data/jrdb/extracted/ and updates
existing DB records with newly parsed fields. No network I/O.

Usage:
    uv run python scripts/keiba_reparse_local.py
    uv run python scripts/keiba_reparse_local.py --types kyi sed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import structlog

from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.keiba.database.repository import KeibaRepository
from providence.keiba.scraper.field_defs.kyi import FIELDS as KYI_FIELDS
from providence.keiba.scraper.field_defs.kyi import RECORD_LENGTH as KYI_LEN
from providence.keiba.scraper.field_defs.sed import FIELDS as SED_FIELDS
from providence.keiba.scraper.field_defs.sed import RECORD_LENGTH as SED_LEN
from providence.keiba.scraper.parser import parse_file

logger = structlog.get_logger()

EXTRACTED_DIR = Path("data/jrdb/extracted")


def reparse_kyi(repo: KeibaRepository, session_factory) -> int:
    kyi_dir = EXTRACTED_DIR / "kyi"
    if not kyi_dir.exists():
        logger.warning("kyi_dir_not_found", path=str(kyi_dir))
        return 0

    files = sorted(kyi_dir.glob("KYI*.txt"))
    logger.info("kyi_files_found", count=len(files))

    total = 0
    for i, fpath in enumerate(files):
        data = fpath.read_bytes()
        try:
            records = parse_file(data, KYI_FIELDS, KYI_LEN)
        except Exception:
            logger.exception("parse_failed", file=fpath.name)
            continue

        with session_factory() as session:
            with session.begin():
                saved = repo.save_entries(session, records)
                total += len(records)

        if (i + 1) % 100 == 0:
            logger.info("kyi_progress", processed=i + 1, total_files=len(files))

    logger.info("kyi_complete", total_records=total, files=len(files))
    return total


def reparse_sed(repo: KeibaRepository, session_factory) -> int:
    sed_dir = EXTRACTED_DIR / "sed"
    if not sed_dir.exists():
        logger.warning("sed_dir_not_found", path=str(sed_dir))
        return 0

    files = sorted(sed_dir.glob("SED*.txt"))
    logger.info("sed_files_found", count=len(files))

    total = 0
    for i, fpath in enumerate(files):
        data = fpath.read_bytes()
        try:
            records = parse_file(data, SED_FIELDS, SED_LEN)
        except Exception:
            logger.exception("parse_failed", file=fpath.name)
            continue

        with session_factory() as session:
            with session.begin():
                saved = repo.save_results(session, records)
                total += len(records)

        if (i + 1) % 100 == 0:
            logger.info("sed_progress", processed=i + 1, total_files=len(files))

    logger.info("sed_complete", total_records=total, files=len(files))
    return total


def main():
    parser = argparse.ArgumentParser(description="Re-parse local JRDB files")
    parser.add_argument("--types", nargs="+", default=["kyi", "sed"],
                        help="File types to re-parse (default: kyi sed)")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    settings = get_settings()
    repo = KeibaRepository()
    session_factory = get_session_factory(settings)

    t0 = time.time()
    for ft in args.types:
        if ft == "kyi":
            reparse_kyi(repo, session_factory)
        elif ft == "sed":
            reparse_sed(repo, session_factory)
        else:
            logger.warning("unknown_type", type=ft)

    logger.info("reparse_complete", elapsed=f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
