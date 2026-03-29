"""oddspark.com HTML scraper.

All endpoints are plain GET requests with query parameters — no CSRF needed.
HTML tables are parsed with BeautifulSoup + lxml.
"""

from __future__ import annotations

from datetime import date

from bs4 import BeautifulSoup, Tag

from providence.config import Settings
from providence.domain.enums import (
    EntryStatus,
    RiderRank,
    TicketType,
    TrackCode,
    TrackCondition,
)
from providence.scraper.base import BaseScraper
from providence.scraper.schemas import (
    EntryRow,
    OddsRow,
    PlayerProfileResponse,
    PlayerSummary,
    RaceEntriesResponse,
    RaceResultResponse,
    RefundRow,
    ResultRow,
)

_PLACE_CODE_MAP: dict[TrackCode, str] = {
    TrackCode.KAWAGUCHI: "02",
    TrackCode.ISESAKI: "03",
    TrackCode.HAMAMATSU: "04",
    TrackCode.IIZUKA: "05",
    TrackCode.SANYO: "06",
}

_BET_TYPE_MAP: dict[TicketType, str] = {
    TicketType.WIN: "1",
    TicketType.EXACTA: "5",
    TicketType.QUINELLA: "6",
    TicketType.WIDE: "7",
    TicketType.TRIFECTA: "8",
    TicketType.TRIO: "9",
}


class OddsparkScraper(BaseScraper):
    """oddspark.com HTML scraper for autorace data."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.oddspark_base_url

    def _url(self, path: str, **params: str | int) -> str:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._base_url}/autorace/{path}?{qs}"

    def _date_str(self, d: date) -> str:
        return d.strftime("%Y%m%d")

    def _place_code(self, track: TrackCode) -> str:
        return _PLACE_CODE_MAP[track]

    async def _get_soup(self, path: str, **params: str | int) -> BeautifulSoup:
        url = self._url(path, **params)
        resp = await self._request("GET", url)
        return BeautifulSoup(resp.text, "lxml")

    # ------------------------------------------------------------------ #
    #  Race Results
    # ------------------------------------------------------------------ #

    async def get_race_result(
        self, track: TrackCode, race_date: date, race_no: int
    ) -> RaceResultResponse:
        soup = await self._get_soup(
            "RaceResult.do",
            raceDy=self._date_str(race_date),
            placeCd=self._place_code(track),
            raceNo=race_no,
        )

        results = _parse_result_table(soup)
        refunds = _parse_refund_section(soup)
        weather, track_cond, temp, humid, road_temp = _parse_race_conditions(soup)

        return RaceResultResponse(
            track=track,
            race_date=race_date,
            race_number=race_no,
            weather=weather,
            track_condition=track_cond,
            temperature=temp,
            humidity=humid,
            track_temperature=road_temp,
            results=results,
            refunds=refunds,
        )

    # ------------------------------------------------------------------ #
    #  Odds
    # ------------------------------------------------------------------ #

    async def get_odds(
        self, track: TrackCode, race_date: date, race_no: int, ticket_type: TicketType
    ) -> list[OddsRow]:
        bet_type = _BET_TYPE_MAP.get(ticket_type)
        if not bet_type:
            return []

        soup = await self._get_soup(
            "Odds.do",
            raceDy=self._date_str(race_date),
            placeCd=self._place_code(track),
            raceNo=race_no,
            betType=bet_type,
        )
        return _parse_odds_table(soup, ticket_type)

    # ------------------------------------------------------------------ #
    #  Race Entries
    # ------------------------------------------------------------------ #

    async def get_race_entries(
        self, track: TrackCode, race_date: date, race_no: int
    ) -> RaceEntriesResponse:
        soup = await self._get_soup(
            "RaceList.do",
            raceDy=self._date_str(race_date),
            placeCd=self._place_code(track),
            raceNo=race_no,
        )

        entries = _parse_entry_table(soup)
        title = _extract_text(soup, "h3")
        weather, track_cond, temp, humid, road_temp = _parse_race_conditions(soup)

        return RaceEntriesResponse(
            track=track,
            race_date=race_date,
            race_number=race_no,
            title=title,
            weather=weather,
            track_condition=track_cond,
            temperature=temp,
            humidity=humid,
            track_temperature=road_temp,
            entries=entries,
        )

    # ------------------------------------------------------------------ #
    #  Race Conditions (weather, track)
    # ------------------------------------------------------------------ #

    async def get_race_conditions(
        self, track: TrackCode, race_date: date
    ) -> dict[str, object]:
        """Fetch weather/track conditions for a day at a track.

        Returns dict with keys: weather, track_condition, temperature,
        humidity, track_temperature. Values are None if not found.
        Conditions are per-day (same for all races on that day at that track).
        """
        soup = await self._get_soup(
            "RaceResult.do",
            raceDy=self._date_str(race_date),
            placeCd=self._place_code(track),
            raceNo=1,
        )
        return _parse_conditions_block(soup)

    # ------------------------------------------------------------------ #
    #  Daily Refunds
    # ------------------------------------------------------------------ #

    async def get_daily_refund(self, track: TrackCode, race_date: date) -> list[RefundRow]:
        soup = await self._get_soup(
            "RaceRefund.do",
            raceDy=self._date_str(race_date),
            placeCd=self._place_code(track),
        )
        return _parse_daily_refund(soup)

    # ------------------------------------------------------------------ #
    #  Players
    # ------------------------------------------------------------------ #

    async def get_all_players(self) -> list[PlayerSummary]:
        """Fetch all riders from RecordRanking (paginated, ~420 players)."""
        all_players: list[PlayerSummary] = []
        page = 1

        while True:
            soup = await self._get_soup("RecordRanking.do", page=page)
            players = _parse_ranking_table(soup)
            if not players:
                break
            all_players.extend(players)
            page += 1

        return all_players

    async def get_player_detail(self, player_code: str) -> PlayerProfileResponse:
        soup = await self._get_soup("PlayerDetail.do", playerCd=player_code)
        return _parse_player_detail(soup, player_code)


# ------------------------------------------------------------------ #
#  HTML Parsing helpers
# ------------------------------------------------------------------ #


def _safe_float(text: str | None) -> float | None:
    if not text:
        return None
    text = text.strip().replace(",", "")
    if not text or text == "-" or text == "---":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(text: str | None) -> int | None:
    if not text:
        return None
    text = text.strip().replace(",", "")
    if not text or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _extract_text(soup: BeautifulSoup | Tag, selector: str) -> str | None:
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else None


def _get_cells(row: Tag) -> list[str]:
    return [td.get_text(strip=True) for td in row.find_all(["td", "th"])]


def _parse_track_condition(text: str | None) -> TrackCondition | None:
    if not text:
        return None
    for tc in TrackCondition:
        if tc.value in text:
            return tc
    return None


def _parse_rank(text: str | None) -> RiderRank | None:
    if not text:
        return None
    for r in RiderRank:
        if r.value == text.strip():
            return r
    return None


def _parse_conditions_block(soup: BeautifulSoup) -> dict[str, object]:
    """Parse the weather/conditions info block from oddspark RaceResult page.

    The block contains text like:
        天候：晴  走路状況：良走路  走路温度：37.0℃  気温：15.0℃  湿度：35.0%
    """
    result: dict[str, object] = {
        "weather": None,
        "track_condition": None,
        "temperature": None,
        "humidity": None,
        "track_temperature": None,
    }

    full_text = soup.get_text()

    import re

    weather_m = re.search(r"天候[：:](\S+)", full_text)
    if weather_m:
        result["weather"] = weather_m.group(1).strip()

    track_m = re.search(r"走路状況[：:](\S+)", full_text)
    if track_m:
        raw = track_m.group(1).strip().replace("走路", "")
        result["track_condition"] = _parse_track_condition(raw)

    temp_m = re.search(r"気温[：:](\d+\.?\d*)℃", full_text)
    if temp_m:
        result["temperature"] = float(temp_m.group(1))

    humid_m = re.search(r"湿度[：:](\d+\.?\d*)%", full_text)
    if humid_m:
        result["humidity"] = float(humid_m.group(1))

    road_m = re.search(r"走路温度[：:](\d+\.?\d*)℃", full_text)
    if road_m:
        result["track_temperature"] = float(road_m.group(1))

    return result


def _parse_race_conditions(soup: BeautifulSoup) -> tuple[
    str | None, TrackCondition | None, float | None, float | None, float | None
]:
    """Extract weather/track conditions from page text."""
    weather = None
    track_cond = None
    temp = None
    humid = None
    road_temp = None

    for text_block in soup.stripped_strings:
        text = str(text_block)
        if "天候" in text and ":" in text:
            weather = text.split(":")[-1].strip()
        elif "走路" in text and ":" in text:
            track_cond = _parse_track_condition(text.split(":")[-1].strip())
        elif "気温" in text:
            temp = _safe_float(text.replace("気温", "").replace("℃", "").strip())
        elif "湿度" in text:
            humid = _safe_float(text.replace("湿度", "").replace("%", "").strip())
        elif "走路温度" in text:
            road_temp = _safe_float(text.replace("走路温度", "").replace("℃", "").strip())

    return weather, track_cond, temp, humid, road_temp


def _parse_result_table(soup: BeautifulSoup) -> list[ResultRow]:
    results: list[ResultRow] = []
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = _get_cells(row)
            if len(cells) < 8:
                continue

            pos = _safe_int(cells[0])
            accident = cells[1].strip() if len(cells) > 1 else None
            car_no = _safe_int(cells[2])
            if car_no is None:
                continue

            player_link = row.find("a", href=True)
            reg_number = ""
            if player_link and "playerCd" in str(player_link.get("href", "")):
                href = str(player_link["href"])
                if "playerCd=" in href:
                    reg_number = href.split("playerCd=")[-1].split("&")[0]
            name = cells[3] if len(cells) > 3 else ""

            results.append(ResultRow(
                post_position=car_no,
                rider_registration_number=reg_number or name,
                finish_position=pos,
                race_time=_safe_float(cells[8]) if len(cells) > 8 else None,
                trial_time=_safe_float(cells[7]) if len(cells) > 7 else None,
                start_timing=_safe_float(cells[9]) if len(cells) > 9 else None,
                accident_code=accident if accident and accident != "-" else None,
                entry_status=_entry_status_from_accident(accident),
            ))

        if results:
            break

    return results


def _entry_status_from_accident(accident: str | None) -> EntryStatus:
    if not accident or accident == "-" or accident == "":
        return EntryStatus.RACING
    if "落" in accident:
        return EntryStatus.FELL
    if "取" in accident or "欠" in accident:
        return EntryStatus.CANCELLED
    if "失" in accident:
        return EntryStatus.DISQUALIFIED
    return EntryStatus.RACING


def _parse_refund_section(soup: BeautifulSoup) -> list[RefundRow]:
    refunds: list[RefundRow] = []

    ticket_name_map: dict[str, TicketType] = {
        "単勝": TicketType.WIN,
        "複勝": TicketType.PLACE,
        "2連複": TicketType.QUINELLA,
        "2連単": TicketType.EXACTA,
        "ワイド": TicketType.WIDE,
        "3連複": TicketType.TRIO,
        "3連単": TicketType.TRIFECTA,
    }

    tables = soup.find_all("table")
    for table in tables:
        header_text = ""
        prev = table.find_previous(["h3", "h4", "caption", "th"])
        if prev:
            header_text = prev.get_text(strip=True)

        ticket_type = None
        for name, tt in ticket_name_map.items():
            if name in header_text or name in table.get_text():
                ticket_type = tt
                break

        if ticket_type is None:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cells = _get_cells(row)
            if len(cells) < 2:
                continue

            combo = cells[0].replace("→", "-").replace("−", "-").replace("ー", "-").strip()
            amount = _safe_int(cells[1].replace("円", "").replace("¥", ""))
            pop = _safe_int(cells[2]) if len(cells) > 2 else None

            if combo and amount and amount > 0:
                refunds.append(RefundRow(
                    ticket_type=ticket_type,
                    combination=combo,
                    refund_amount=amount,
                    popularity=pop,
                ))

    return refunds


def _parse_daily_refund(soup: BeautifulSoup) -> list[RefundRow]:
    """Parse the daily refund summary page (RaceRefund.do)."""
    return _parse_refund_section(soup)


def _parse_entry_table(soup: BeautifulSoup) -> list[EntryRow]:
    entries: list[EntryRow] = []
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = _get_cells(row)
            if len(cells) < 6:
                continue

            car_no = _safe_int(cells[0])
            if car_no is None or car_no < 1 or car_no > 8:
                continue

            player_link = row.find("a", href=True)
            reg_number = ""
            if player_link and "playerCd" in str(player_link.get("href", "")):
                href = str(player_link["href"])
                reg_number = href.split("playerCd=")[-1].split("&")[0]

            entries.append(EntryRow(
                post_position=car_no,
                rider_registration_number=reg_number or cells[1],
                rider_name=cells[1],
                handicap_meters=_safe_int(cells[5]) or 0,
                trial_time=_safe_float(cells[6]) if len(cells) > 6 else None,
                avg_trial_time=_safe_float(cells[7]) if len(cells) > 7 else None,
                race_score=_safe_float(cells[4]) if len(cells) > 4 else None,
            ))

        if entries:
            break

    return entries


def _parse_odds_table(soup: BeautifulSoup, ticket_type: TicketType) -> list[OddsRow]:
    odds: list[OddsRow] = []
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = _get_cells(row)
            if len(cells) < 2:
                continue

            combo = cells[0].replace("→", "-").replace("−", "-").replace("ー", "-").strip()
            odds_val = _safe_float(cells[1])

            if combo and odds_val is not None and odds_val > 0:
                pop = _safe_int(cells[2]) if len(cells) > 2 else None
                odds.append(OddsRow(
                    ticket_type=ticket_type,
                    combination=combo,
                    odds_value=odds_val,
                    popularity=pop,
                ))

        if odds:
            break

    return odds


def _parse_ranking_table(soup: BeautifulSoup) -> list[PlayerSummary]:
    players: list[PlayerSummary] = []
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cells = _get_cells(row)
            if len(cells) < 6:
                continue

            player_link = row.find("a", href=True)
            reg_number = ""
            if player_link and "playerCd" in str(player_link.get("href", "")):
                href = str(player_link["href"])
                reg_number = href.split("playerCd=")[-1].split("&")[0]

            name = cells[1] if len(cells) > 1 else ""
            if not reg_number and not name:
                continue

            players.append(PlayerSummary(
                registration_number=reg_number or name,
                name=name,
                age=_safe_int(cells[2]) if len(cells) > 2 else None,
                generation=_safe_int(cells[3]) if len(cells) > 3 else None,
                rank=_parse_rank(cells[5]) if len(cells) > 5 else None,
            ))

        if players:
            break

    return players


def _parse_player_detail(soup: BeautifulSoup, player_code: str) -> PlayerProfileResponse:
    name = _extract_text(soup, "h2") or ""
    name_kana = None
    birth_year = None
    generation = None
    rank = None
    home_track = None

    for text in soup.stripped_strings:
        text_s = str(text)
        if "期別" in text_s:
            generation = _safe_int(text_s.replace("期別", "").replace("期", "").strip())
        elif "ランク" in text_s:
            for r in RiderRank:
                if r.value in text_s:
                    rank = r
                    break
        elif "LG" in text_s or "所属" in text_s:
            for tc in TrackCode:
                if tc.japanese_name in text_s:
                    home_track = tc
                    break

    return PlayerProfileResponse(
        registration_number=player_code,
        name=name,
        name_kana=name_kana,
        birth_year=birth_year,
        generation=generation,
        home_track=home_track,
        rank=rank,
    )
