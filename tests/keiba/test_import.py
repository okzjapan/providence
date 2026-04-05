"""Tests for the keiba import pipeline."""


from scripts.keiba_import import IMPORT_ORDER, _extract_payouts


class TestImportOrder:
    def test_fk_safe_order(self):
        """Masters must come before entities that reference them."""
        entity_order = [entry[5] for entry in IMPORT_ORDER]
        assert entity_order == [
            "jockeys",   # no FK deps
            "trainers",  # no FK deps
            "horses",    # self-ref FK (nullable)
            "races",     # FK → racecourses
            "entries",   # FK → races, horses, jockeys, trainers
            "results",   # FK → entries
            "payouts",   # FK → races
        ]


class TestExtractPayouts:
    def test_win_payouts(self):
        rec = {
            "win_horse_1": 3, "win_payout_1": 500,
            "win_horse_2": None, "win_payout_2": None,
            "win_horse_3": None, "win_payout_3": None,
        }
        payouts = _extract_payouts(rec)
        win_payouts = [p for p in payouts if p["ticket_type"] == "win"]
        assert len(win_payouts) == 1
        assert win_payouts[0]["combination"] == "3"
        assert win_payouts[0]["payout_amount"] == 500

    def test_skips_zero_payouts(self):
        rec = {
            "win_horse_1": 3, "win_payout_1": 0,
        }
        payouts = _extract_payouts(rec)
        assert len([p for p in payouts if p["ticket_type"] == "win"]) == 0

    def test_combo_payouts(self):
        rec = {
            "quinella_combo_1": "0305", "quinella_payout_1": 1200,
            "quinella_combo_2": None, "quinella_payout_2": None,
            "quinella_combo_3": None, "quinella_payout_3": None,
        }
        payouts = _extract_payouts(rec)
        q_payouts = [p for p in payouts if p["ticket_type"] == "quinella"]
        assert len(q_payouts) == 1
        assert q_payouts[0]["combination"] == "0305"
