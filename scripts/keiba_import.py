#!/usr/bin/env python3
"""JRDB data import pipeline.

Downloads, parses, and stores JRDB data into the local database.
Supports both yearly bulk imports and single-day imports.

Usage:
    # Bulk import 2015-2025
    uv run python scripts/keiba_import.py --start-year 2015 --end-year 2025

    # Import a single day
    uv run python scripts/keiba_import.py --date 260404

    # Force re-import (ignore import log)
    uv run python scripts/keiba_import.py --start-year 2024 --end-year 2024 --force
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import structlog

from providence.config import get_settings
from providence.database.engine import get_session_factory
from providence.keiba.database.repository import KeibaRepository
from providence.keiba.scraper.field_defs.bac import FIELDS as BAC_FIELDS
from providence.keiba.scraper.field_defs.bac import RECORD_LENGTH as BAC_LEN
from providence.keiba.scraper.field_defs.hjc import FIELDS as HJC_FIELDS
from providence.keiba.scraper.field_defs.hjc import RECORD_LENGTH as HJC_LEN
from providence.keiba.scraper.field_defs.kyi import FIELDS as KYI_FIELDS
from providence.keiba.scraper.field_defs.kyi import RECORD_LENGTH as KYI_LEN
from providence.keiba.scraper.field_defs.master import (
    CSA_FIELDS,
    CSA_RECORD_LENGTH,
    KSA_FIELDS,
    KSA_RECORD_LENGTH,
)
from providence.keiba.scraper.field_defs.sed import FIELDS as SED_FIELDS
from providence.keiba.scraper.field_defs.sed import RECORD_LENGTH as SED_LEN
from providence.keiba.scraper.field_defs.ukc import FIELDS as UKC_FIELDS
from providence.keiba.scraper.field_defs.ukc import RECORD_LENGTH as UKC_LEN
from providence.keiba.scraper.jrdb import JrdbDownloader
from providence.keiba.scraper.parser import parse_file

logger = structlog.get_logger()

# FK-safe import order: masters first, then races, entries, results, payouts
# (jrdb_dir, file_prefix, file_type, fields, record_length, entity_type)
IMPORT_ORDER = [
    ("Ks", "KSA", "ksa", KSA_FIELDS, KSA_RECORD_LENGTH, "jockeys"),
    ("Cs", "CSA", "csa", CSA_FIELDS, CSA_RECORD_LENGTH, "trainers"),
    ("Ukc", "UKC", "ukc", UKC_FIELDS, UKC_LEN, "horses"),
    ("Bac", "BAC", "bac", BAC_FIELDS, BAC_LEN, "races"),
    ("Kyi", "KYI", "kyi", KYI_FIELDS, KYI_LEN, "entries"),
    ("Sed", "SED", "sed", SED_FIELDS, SED_LEN, "results"),
    ("Hjc", "HJC", "hjc", HJC_FIELDS, HJC_LEN, "payouts"),
]


def _save_records(
    repo: KeibaRepository,
    session,
    entity_type: str,
    records: list[dict[str, Any]],
    file_type: str,
) -> int:
    if entity_type == "jockeys":
        return repo.save_jockeys(session, records)
    if entity_type == "trainers":
        return repo.save_trainers(session, records)
    if entity_type == "horses":
        return repo.save_horses(session, records)
    if entity_type == "races":
        return repo.save_races(session, records)
    if entity_type == "entries":
        return repo.save_entries(session, records)
    if entity_type == "results":
        return repo.save_results(session, records)
    if entity_type == "payouts":
        return _save_payouts_from_hjc(repo, session, records)
    return 0


def _save_payouts_from_hjc(
    repo: KeibaRepository,
    session,
    records: list[dict[str, Any]],
) -> int:
    """Convert HJC flat records into per-race payout lists and save."""
    total = 0
    for rec in records:
        race_key = repo._build_race_key(rec)
        if not race_key:
            continue
        payouts = _extract_payouts(rec)
        total += repo.save_payouts(session, race_key, payouts)
    return total


def _extract_payouts(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract individual payout entries from a flat HJC record."""
    payouts = []
    _add_payouts(payouts, rec, "win", "win_horse_{i}", "win_payout_{i}", 3)
    _add_payouts(payouts, rec, "place", "place_horse_{i}", "place_payout_{i}", 5)
    _add_combo_payouts(payouts, rec, "bracket_quinella", "bracket_quinella_combo_{i}", "bracket_quinella_payout_{i}", 1)
    _add_combo_payouts(payouts, rec, "quinella", "quinella_combo_{i}", "quinella_payout_{i}", 3)
    _add_combo_payouts(payouts, rec, "wide", "wide_combo_{i}", "wide_payout_{i}", 3)
    _add_combo_payouts(payouts, rec, "exacta", "exacta_combo_{i}", "exacta_payout_{i}", 3)
    _add_combo_payouts(payouts, rec, "trio", "trio_combo_{i}", "trio_payout_{i}", 3)
    _add_combo_payouts(payouts, rec, "trifecta", "trifecta_combo_{i}", "trifecta_payout_{i}", 6)
    return payouts


def _add_payouts(payouts, rec, ticket_type, horse_key, payout_key, count):
    for i in range(1, count + 1):
        horse = rec.get(horse_key.format(i=i))
        payout = rec.get(payout_key.format(i=i))
        if horse and payout and payout > 0:
            payouts.append({"ticket_type": ticket_type, "combination": str(horse), "payout_amount": payout})


def _add_combo_payouts(payouts, rec, ticket_type, combo_key, payout_key, count):
    for i in range(1, count + 1):
        combo = rec.get(combo_key.format(i=i))
        payout = rec.get(payout_key.format(i=i))
        if combo and payout and payout > 0:
            payouts.append({"ticket_type": ticket_type, "combination": combo.strip(), "payout_amount": payout})


def import_yearly(
    downloader: JrdbDownloader,
    repo: KeibaRepository,
    session_factory,
    start_year: int,
    end_year: int,
    force: bool = False,
    strict: bool = False,
) -> dict[str, int]:
    totals: dict[str, int] = {}

    with session_factory() as session:
        with session.begin():
            repo.ensure_racecourses(session)

    for jrdb_dir, prefix, file_type, fields, record_length, entity_type in IMPORT_ORDER:
        for year in range(start_year, end_year + 1):
            file_label = f"{prefix}_{year}"
            log = logger.bind(file_type=file_type, year=year)

            with session_factory() as session:
                if not force and repo.is_imported(session, file_label):
                    log.info("skipping_already_imported", file=file_label)
                    continue

            log.info("downloading", file=file_label)
            try:
                extract_dir = downloader.download_yearly(jrdb_dir, prefix, year)
            except Exception:
                log.exception("download_failed", file=file_label)
                if strict:
                    raise
                continue

            if extract_dir is None:
                log.warning("not_available", file=file_label)
                continue

            files = downloader.read_extracted_files(jrdb_dir.lower())
            total_records = 0
            for fname, data in files:
                try:
                    records = parse_file(data, fields, record_length)
                except Exception:
                    log.exception("parse_failed", file=fname)
                    if strict:
                        raise
                    continue

                with session_factory() as session:
                    with session.begin():
                        saved = _save_records(repo, session, entity_type, records, file_type)
                        total_records += saved

                log.info("imported_file", file=fname, records=len(records), saved=saved)

            with session_factory() as session:
                with session.begin():
                    repo.log_import(session, file_label, file_type, total_records)

            key = f"{file_type}_{year}"
            totals[key] = total_records
            log.info("year_complete", file=file_label, total=total_records)

    return totals


def import_daily(
    downloader: JrdbDownloader,
    repo: KeibaRepository,
    session_factory,
    date_str: str,
    force: bool = False,
) -> dict[str, int]:
    totals: dict[str, int] = {}

    with session_factory() as session:
        with session.begin():
            repo.ensure_racecourses(session)

    for jrdb_dir, prefix, file_type, fields, record_length, entity_type in IMPORT_ORDER:
        file_label = f"{prefix}{date_str}"
        log = logger.bind(file_type=file_type, date=date_str)

        with session_factory() as session:
            if not force and repo.is_imported(session, file_label):
                log.info("skipping_already_imported", file=file_label)
                continue

        log.info("downloading", file=file_label)
        try:
            extract_dir = downloader.download_daily(jrdb_dir, prefix, date_str)
        except Exception:
            log.exception("download_failed", file=file_label)
            continue

        if extract_dir is None:
            log.warning("not_available", file=file_label)
            continue

        files = downloader.read_extracted_files(jrdb_dir.lower())
        total_records = 0
        for fname, data in files:
            try:
                records = parse_file(data, fields, record_length)
            except Exception:
                log.exception("parse_failed", file=fname)
                continue

            with session_factory() as session:
                with session.begin():
                    saved = _save_records(repo, session, entity_type, records, file_type)
                    total_records += saved

            log.info("imported_file", file=fname, records=len(records), saved=saved)

        with session_factory() as session:
            with session.begin():
                repo.log_import(session, file_label, file_type, total_records)

        totals[file_type] = total_records

    return totals


def main():
    parser = argparse.ArgumentParser(description="JRDB data import pipeline")
    parser.add_argument("--start-year", type=int, help="Start year for bulk import")
    parser.add_argument("--end-year", type=int, help="End year for bulk import")
    parser.add_argument("--date", type=str, help="Single day import (yyMMdd format)")
    parser.add_argument("--force", action="store_true", help="Re-import even if already imported")
    parser.add_argument("--strict", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    settings = get_settings()
    if not settings.jrdb_user_id or not settings.jrdb_password:
        logger.error("JRDB credentials not configured in .env")
        sys.exit(1)

    downloader = JrdbDownloader(settings)
    repo = KeibaRepository()
    session_factory = get_session_factory(settings)

    if args.date:
        totals = import_daily(downloader, repo, session_factory, args.date, force=args.force)
    elif args.start_year and args.end_year:
        totals = import_yearly(
            downloader, repo, session_factory,
            args.start_year, args.end_year,
            force=args.force, strict=args.strict,
        )
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("import_complete", totals=totals)


if __name__ == "__main__":
    main()
