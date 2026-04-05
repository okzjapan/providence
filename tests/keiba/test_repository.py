"""Tests for KeibaRepository."""

import pytest

from providence.keiba.database.repository import KeibaRepository
from providence.keiba.database.tables import (
    KeibaHorse,
    KeibaJockey,
    KeibaRace,
    KeibaRacecourse,
    KeibaRaceEntry,
    KeibaRaceResult,
    KeibaTicketPayout,
)


@pytest.fixture()
def repo():
    return KeibaRepository()


class TestEnsureRacecourses:
    def test_creates_10_racecourses(self, session_factory, repo):
        with session_factory() as session:
            with session.begin():
                repo.ensure_racecourses(session)
            courses = session.query(KeibaRacecourse).all()
            assert len(courses) == 10
            assert courses[0].name == "札幌"
            assert courses[4].name == "東京"

    def test_idempotent(self, session_factory, repo):
        with session_factory() as session:
            with session.begin():
                repo.ensure_racecourses(session)
                repo.ensure_racecourses(session)
            assert session.query(KeibaRacecourse).count() == 10


class TestSaveJockeys:
    def test_insert_new(self, session_factory, repo):
        records = [{"jockey_code": "00001", "jockey_name": "テスト騎手", "affiliation_code": 2}]
        with session_factory() as session:
            with session.begin():
                count = repo.save_jockeys(session, records)
            assert count == 1
            j = session.query(KeibaJockey).first()
            assert j.jockey_code == "00001"
            assert j.jockey_name == "テスト騎手"

    def test_upsert_existing(self, session_factory, repo):
        records = [{"jockey_code": "00001", "jockey_name": "旧名"}]
        with session_factory() as session:
            with session.begin():
                repo.save_jockeys(session, records)
                repo.save_jockeys(session, [{"jockey_code": "00001", "jockey_name": "新名"}])
            j = session.query(KeibaJockey).first()
            assert j.jockey_name == "新名"
            assert session.query(KeibaJockey).count() == 1


class TestSaveHorses:
    def test_insert_with_nullable_bloodline_fk(self, session_factory, repo):
        records = [{"blood_registration_number": "12345678", "horse_name": "テスト馬", "sex_code": 1}]
        with session_factory() as session:
            with session.begin():
                count = repo.save_horses(session, records)
            assert count == 1
            h = session.query(KeibaHorse).first()
            assert h.sire_id is None
            assert h.dam_id is None


class TestSaveRaces:
    def test_insert_race(self, session_factory, repo):
        records = [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "race_date": "20250101", "distance": 2000, "surface_code": 1, "num_runners": 16,
            "race_name": "テストレース",
        }]
        with session_factory() as session:
            with session.begin():
                repo.ensure_racecourses(session)
                count = repo.save_races(session, records)
            assert count == 1
            race = session.query(KeibaRace).first()
            assert race.race_key == "05251101"
            assert race.distance == 2000


class TestSaveEntries:
    def _setup_race(self, session, repo):
        repo.ensure_racecourses(session)
        repo.save_races(session, [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "race_date": "20250101", "distance": 2000, "surface_code": 1,
        }])

    def test_insert_entry(self, session_factory, repo):
        records = [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "post_position": 3, "blood_registration_number": "12345678",
            "idm": 55.0, "impost_weight": 57.0,
        }]
        with session_factory() as session:
            with session.begin():
                self._setup_race(session, repo)
                count = repo.save_entries(session, records)
            assert count == 1
            entry = session.query(KeibaRaceEntry).first()
            assert entry.post_position == 3
            assert entry.idm == 55.0


class TestSaveResults:
    def _setup_race_and_entry(self, session, repo):
        repo.ensure_racecourses(session)
        repo.save_races(session, [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "race_date": "20250101", "distance": 2000, "surface_code": 1,
        }])
        repo.save_entries(session, [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "post_position": 3,
        }])

    def test_insert_result(self, session_factory, repo):
        records = [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "post_position": 3, "finish_position": 1, "race_time": 120.5, "last_3f_time": 34.2,
        }]
        with session_factory() as session:
            with session.begin():
                self._setup_race_and_entry(session, repo)
                count = repo.save_results(session, records)
            assert count == 1
            result = session.query(KeibaRaceResult).first()
            assert result.finish_position == 1


class TestSavePayouts:
    def _setup_race(self, session, repo):
        repo.ensure_racecourses(session)
        repo.save_races(session, [{
            "place_code": "05", "year": "25", "kai": "1", "day": 1, "race_number": "01",
            "race_date": "20250101", "distance": 2000, "surface_code": 1,
        }])

    def test_insert_payout(self, session_factory, repo):
        payouts = [{"ticket_type": "win", "combination": "03", "payout_amount": 500}]
        with session_factory() as session:
            with session.begin():
                self._setup_race(session, repo)
                count = repo.save_payouts(session, "05251101", payouts)
            assert count == 1
            p = session.query(KeibaTicketPayout).first()
            assert p.payout_amount == 500


class TestImportLog:
    def test_log_and_check(self, session_factory, repo):
        with session_factory() as session:
            with session.begin():
                assert not repo.is_imported(session, "SED_2024.lzh")
                repo.log_import(session, "SED_2024.lzh", "sed", 5000)
                assert repo.is_imported(session, "SED_2024.lzh")
