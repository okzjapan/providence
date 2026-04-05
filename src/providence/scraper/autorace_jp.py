"""autorace.jp JSON API scraper.

All POST endpoints require CSRF token + session cookie.
The GET endpoint /race_info/XML/Hold/Today does not require CSRF.
"""

from __future__ import annotations

import re
from datetime import date

import httpx

from providence.config import Settings
from providence.domain.enums import (
    EntryStatus,
    Grade,
    RiderRank,
    TicketType,
    TrackCode,
    TrackCondition,
)
from providence.scraper.base import BaseScraper
from providence.scraper.schemas import (
    EntryRow,
    OddsResponse,
    OddsRow,
    PlayerProfileResponse,
    PlayerSummary,
    RaceEntriesResponse,
    RaceResultResponse,
    RefundRow,
    ResultRow,
)

_CSRF_PATTERN = re.compile(r'<meta\s+name="csrf-token"\s+content="([^"]+)"')

_GRADE_MAP: dict[int, Grade] = {
    0: Grade.NORMAL,
    1: Grade.GII,
    2: Grade.GI,
    3: Grade.SG,
}

_REFUND_KEYS: list[tuple[str, TicketType]] = [
    ("tns", TicketType.WIN),
    ("fns", TicketType.PLACE),
    ("rtw", TicketType.EXACTA),
    ("rfw", TicketType.QUINELLA),
    ("wid", TicketType.WIDE),
    ("rt3", TicketType.TRIFECTA),
    ("rf3", TicketType.TRIO),
]

# From autorace.jp race.js `situationList` / `situationNameList`
_SITUATION_CODE_MAP: dict[int, TrackCondition | None] = {
    0: TrackCondition.GOOD,   # 良走路
    1: TrackCondition.WET,    # 湿走路
    2: None,                  # 風 (rare; not a surface condition)
    3: None,                  # オイル (rare)
    4: None,                  # 荒 (rare)
    5: TrackCondition.MIXED,  # 斑走路
}

_SITUATION_LABEL: dict[int, str] = {
    0: "良", 1: "湿", 2: "風", 3: "油", 4: "荒", 5: "斑",
}


def _parse_track_condition(value: str | None) -> TrackCondition | None:
    if not value:
        return None
    for tc in TrackCondition:
        if tc.value in value:
            return tc
    return None


def _parse_rank(value: str | None) -> RiderRank | None:
    if not value:
        return None
    for r in RiderRank:
        if r.value == value:
            return r
    return None


class AutoraceJpScraper(BaseScraper):
    """autorace.jp JSON API client with reactive CSRF management."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.autorace_jp_base_url
        self._csrf_token: str | None = None

    async def _fetch_csrf_token(self) -> None:
        resp = await self._request("GET", f"{self._base_url}/race_info/")
        match = _CSRF_PATTERN.search(resp.text)
        if not match:
            raise RuntimeError("Failed to extract CSRF token from autorace.jp")
        self._csrf_token = match.group(1)
        self._log.info("csrf_token_acquired")

    async def _post_api(self, endpoint: str, payload: dict) -> dict:
        """POST with CSRF; refreshes token on 403/419."""
        if not self._csrf_token:
            await self._fetch_csrf_token()
        try:
            return await self._post_with_csrf(endpoint, payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 419):
                self._log.info("csrf_token_expired_refreshing")
                await self._fetch_csrf_token()
                return await self._post_with_csrf(endpoint, payload)
            raise

    async def _post_with_csrf(self, endpoint: str, payload: dict) -> dict:
        headers = {
            "X-CSRF-TOKEN": self._csrf_token or "",
            "X-Requested-With": "XMLHttpRequest",
        }
        resp = await self._request(
            "POST",
            f"{self._base_url}/race_info/{endpoint}",
            json=payload,
            headers=headers,
        )
        return resp.json()

    # ------------------------------------------------------------------ #
    #  Public API methods
    # ------------------------------------------------------------------ #

    async def get_today_schedule(self) -> dict:
        """Today's race schedule (no CSRF required)."""
        resp = await self._request("GET", f"{self._base_url}/race_info/XML/Hold/Today")
        return resp.json()

    async def get_live_conditions(self, track: TrackCode) -> dict[str, object] | None:
        """Fetch real-time weather/track conditions from Today API.

        Returns dict with keys matching ``repo.update_race_conditions`` input,
        or None if the track is not racing today.
        """
        data = await self.get_today_schedule()
        body = data.get("body", data) if isinstance(data, dict) else {}
        for item in body.get("today", []):
            if item.get("placeCode") != track.value:
                continue
            return parse_today_conditions(item)
        return None

    async def get_race_entries(self, track: TrackCode, race_date: date, race_no: int) -> RaceEntriesResponse:
        """Fetch race entries from Program API.

        Note: Program API does NOT return raceInfo (weather, track condition etc.)
        or avgTrialTime / raceScore. Those must come from other sources (Today API
        for current races, oddspark for historical).
        """
        data = await self._post_api("Program", {
            "placeCode": track.value,
            "raceDate": str(race_date),
            "raceNo": race_no,
        })
        body = data.get("body", data) if isinstance(data, dict) else {}
        players = body.get("playerList", []) if isinstance(body, dict) else []

        entries = [
            EntryRow(
                post_position=p.get("carNo", 0),
                rider_registration_number=str(p.get("playerCode", "")),
                rider_name=p.get("playerName", ""),
                age=p.get("age"),
                generation=p.get("graduationCode"),
                rank=_parse_rank(p.get("rank")),
                handicap_meters=p.get("handicap", 0),
                trial_time=_safe_float(p.get("trialRunTime")),
                trial_deviation=_parse_race_dev(p.get("raceDev")),
            )
            for p in players
        ]

        return RaceEntriesResponse(
            track=track,
            race_date=race_date,
            race_number=race_no,
            entries=entries,
        )

    async def get_odds(self, track: TrackCode, race_date: date, race_no: int) -> OddsResponse:
        """Fetch all odds for a race.

        API response format (nested dicts, not flat lists):
          tnsOddsList: {"carNo": "odds_str", ...}           -- 単勝
          fnsOddsList: {"carNo": {"min": "x", "max": "y"}}  -- 複勝 (min-max range)
          rtwOddsList: {"1st": {"2nd": "odds_str", ...}}    -- 2連単
          rfwOddsList: {"low": {"high": "odds_str", ...}}   -- 2連複
          widOddsList: {"low": {"high": {"min","max"}}}     -- ワイド (min-max range)
          rt3OddsList: {"1st": {"2nd": {"3rd": "odds_str"}}}-- 3連単
          rf3OddsList: {"low": {"mid": {"high": "odds_str"}}}-- 3連複
        """
        data = await self._post_api("Odds", {
            "placeCode": track.value,
            "raceDate": str(race_date),
            "raceNo": race_no,
        })
        body = data.get("body", data) if isinstance(data, dict) else {}
        if not isinstance(body, dict):
            return OddsResponse(track=track, race_date=race_date, race_number=race_no, odds=[])

        odds_rows: list[OddsRow] = []

        # 単勝: {carNo: oddsStr}
        for car, odds_str in body.get("tnsOddsList", {}).items():
            val = _safe_float(odds_str)
            if val and val > 0:
                odds_rows.append(OddsRow(ticket_type=TicketType.WIN, combination=car, odds_value=val))

        # 2連単: {1st: {2nd: oddsStr}}
        for first, seconds in body.get("rtwOddsList", {}).items():
            if isinstance(seconds, dict):
                for second, odds_str in seconds.items():
                    val = _safe_float(odds_str)
                    if val and val > 0:
                        odds_rows.append(OddsRow(
                            ticket_type=TicketType.EXACTA, combination=f"{first}-{second}", odds_value=val,
                        ))

        # 2連複: {low: {high: oddsStr}}
        for low, highs in body.get("rfwOddsList", {}).items():
            if isinstance(highs, dict):
                for high, odds_str in highs.items():
                    val = _safe_float(odds_str)
                    if val and val > 0:
                        odds_rows.append(OddsRow(
                            ticket_type=TicketType.QUINELLA, combination=f"{low}-{high}", odds_value=val,
                        ))

        # ワイド: {low: {high: {min, max}}} — use min odds
        for low, highs in body.get("widOddsList", {}).items():
            if isinstance(highs, dict):
                for high, minmax in highs.items():
                    if isinstance(minmax, dict):
                        val = _safe_float(minmax.get("min"))
                    else:
                        val = _safe_float(minmax)
                    if val and val > 0:
                        odds_rows.append(OddsRow(
                            ticket_type=TicketType.WIDE, combination=f"{low}-{high}", odds_value=val,
                        ))

        # 3連単: {1st: {2nd: {3rd: oddsStr}}}
        for first, seconds in body.get("rt3OddsList", {}).items():
            if isinstance(seconds, dict):
                for second, thirds in seconds.items():
                    if isinstance(thirds, dict):
                        for third, odds_str in thirds.items():
                            val = _safe_float(odds_str)
                            if val and val > 0:
                                odds_rows.append(OddsRow(
                                    ticket_type=TicketType.TRIFECTA,
                                    combination=f"{first}-{second}-{third}",
                                    odds_value=val,
                                ))

        # 3連複: {low: {mid: {high: oddsStr}}}
        for low, mids in body.get("rf3OddsList", {}).items():
            if isinstance(mids, dict):
                for mid, highs in mids.items():
                    if isinstance(highs, dict):
                        for high, odds_str in highs.items():
                            val = _safe_float(odds_str)
                            if val and val > 0:
                                odds_rows.append(OddsRow(
                                    ticket_type=TicketType.TRIO,
                                    combination=f"{low}-{mid}-{high}",
                                    odds_value=val,
                                ))

        return OddsResponse(
            track=track,
            race_date=race_date,
            race_number=race_no,
            odds=odds_rows,
        )

    async def get_race_result(self, track: TrackCode, race_date: date, race_no: int) -> RaceResultResponse:
        data = await self._post_api("RaceResult", {
            "placeCode": track.value,
            "raceDate": str(race_date),
            "raceNo": race_no,
        })
        body = data.get("body", data) if isinstance(data, dict) else {}
        result_list = body.get("raceResult", []) if isinstance(body, dict) else []
        refund_info = body.get("refundInfo", {}) if isinstance(body, dict) else {}

        results = [
            ResultRow(
                post_position=r.get("carNo", 0),
                rider_registration_number=str(r.get("playerCode", "")),
                finish_position=r.get("order"),
                race_time=_safe_float(r.get("raceTime")),
                trial_time=_safe_float(r.get("traialTime")),
                start_timing=_safe_float(r.get("st")),
                accident_code=str(r["accidentCode"]) if r.get("accidentCode") is not None else None,
                entry_status=_parse_entry_status(
                    str(r["accidentCode"]) if r.get("accidentCode") is not None else None,
                    str(r["foulCode"]) if r.get("foulCode") is not None else None,
                ),
            )
            for r in result_list
        ]

        refunds = _parse_refunds(refund_info)

        return RaceResultResponse(
            track=track,
            race_date=race_date,
            race_number=race_no,
            results=results,
            refunds=refunds,
        )

    async def get_player_profile(self, player_code: str) -> PlayerProfileResponse:
        data = await self._post_api("Profile", {"playerCode": player_code})
        body = data.get("body", data)
        profile = body.get("profile", {})

        birth_year = None
        birthday = profile.get("birthday")
        if birthday and len(birthday) >= 4:
            try:
                birth_year = int(birthday[:4])
            except ValueError:
                pass

        return PlayerProfileResponse(
            registration_number=str(profile.get("playerCode", player_code)),
            name=profile.get("playerName", ""),
            name_kana=profile.get("playerNameKana"),
            birth_year=birth_year,
            generation=profile.get("graduationCode"),
            home_track=_safe_track_code(profile.get("placeCode")),
            rank=_parse_rank(profile.get("rank")),
        )

    async def search_races(
        self,
        tracks: list[TrackCode],
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Search for races in a date range. Returns list of hold dicts.

        Each hold dict has keys like: placeCode, placeName, title, gradeCode,
        gradeName, raceDateList (list of "YYYY-MM-DD" strings).
        Note: API returns body as a list directly, not wrapped in a dict.
        """
        data = await self._post_api("SearchRace", {
            "placeCodeList": [t.value for t in tracks],
            "startDate": str(start_date),
            "endDate": str(end_date),
            "gradeCodeList": [],
            "title": "",
        })
        body = data.get("body", data) if isinstance(data, dict) else data
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            return body.get("holdList", body.get("raceList", []))
        return []

    async def get_players(self, track: TrackCode, race_date: date) -> list[PlayerSummary]:
        """Get all players entered at a track on a given date."""
        data = await self._post_api("Player", {
            "placeCode": track.value,
            "raceDate": str(race_date),
        })
        body = data.get("body", data)
        summaries: list[PlayerSummary] = []

        for rank_group in ("sPlayerList", "aPlayerList", "bPlayerList"):
            for p in body.get(rank_group, []):
                summaries.append(PlayerSummary(
                    registration_number=str(p.get("playerCode", "")),
                    name=p.get("playerName", ""),
                    age=p.get("age"),
                    generation=p.get("graduationCode"),
                    rank=_parse_rank(p.get("rank")),
                    home_track=_safe_track_code(p.get("placeCode")),
                ))

        return summaries


# ------------------------------------------------------------------ #
#  Helper functions
# ------------------------------------------------------------------ #


def parse_today_conditions(item: dict) -> dict[str, object]:
    """Convert a single Today API schedule item into a conditions dict.

    The returned dict is compatible with ``Repository.update_race_conditions``.
    """
    situation_code = item.get("situationCode")
    track_cond: TrackCondition | None = None
    if situation_code is not None:
        try:
            track_cond = _SITUATION_CODE_MAP.get(int(situation_code))
        except (ValueError, TypeError):
            pass

    return {
        "weather": item.get("weather") or None,
        "track_condition": track_cond,
        "temperature": _safe_float(item.get("temp")),
        "humidity": _safe_float(item.get("humid")),
        "track_temperature": _safe_float(item.get("roadtemp")),
    }


def situation_code_label(code: int | None) -> str:
    """Human-readable label for a situationCode value."""
    if code is None:
        return ""
    return _SITUATION_LABEL.get(int(code), "")


def _parse_race_dev(value: object) -> float | None:
    """Parse raceDev field: API returns "059" meaning 0.059 seconds."""
    if value is None or value == "":
        return None
    try:
        raw = str(value).strip()
        if not raw:
            return None
        int_val = int(raw)
        return int_val / 1000.0
    except (ValueError, TypeError):
        return _safe_float(value)


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_track_code(value: object) -> TrackCode | None:
    if value is None:
        return None
    try:
        return TrackCode(int(value))
    except (ValueError, TypeError):
        return None


def _parse_entry_status(accident_code: str | None, foul_code: str | None) -> EntryStatus:
    if accident_code and accident_code != "0":
        code_map = {"1": EntryStatus.FELL, "2": EntryStatus.CANCELLED}
        return code_map.get(accident_code, EntryStatus.RACING)
    if foul_code and foul_code != "0":
        return EntryStatus.DISQUALIFIED
    return EntryStatus.RACING


def _parse_refunds(refund_info: dict) -> list[RefundRow]:
    refunds: list[RefundRow] = []
    for key, ticket_type in _REFUND_KEYS:
        raw = refund_info.get(key)
        if not raw:
            continue
        # API wraps items in {"typeCode": ..., "list": [...]}
        if isinstance(raw, dict):
            items = raw.get("list", [raw])
        elif isinstance(raw, list):
            items = raw
        else:
            continue
        for item in items:
            combo_parts = []
            # 単勝/複勝 use "carNo", others use "1thCarNo"/"2thCarNo"/"3thCarNo"
            if "carNo" in item:
                combo_parts.append(str(item["carNo"]))
            else:
                for pos_key in ("1thCarNo", "2thCarNo", "3thCarNo"):
                    val = item.get(pos_key)
                    if val is not None and val != 0:
                        combo_parts.append(str(val))
            if not combo_parts:
                continue
            combination = "-".join(combo_parts)
            refund_amount = item.get("refund", 0)
            if refund_amount and refund_amount > 0:
                refunds.append(RefundRow(
                    ticket_type=ticket_type,
                    combination=combination,
                    refund_amount=refund_amount,
                    popularity=item.get("pop"),
                ))
    return refunds
