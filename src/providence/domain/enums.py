from enum import Enum, IntEnum


class TrackCode(IntEnum):
    """レース場コード (autorace.jp / oddspark 共通)"""

    KAWAGUCHI = 2
    ISESAKI = 3
    HAMAMATSU = 4
    IIZUKA = 5
    SANYO = 6

    @property
    def japanese_name(self) -> str:
        return _TRACK_NAMES[self]

    @classmethod
    def from_name(cls, name: str) -> "TrackCode":
        for code, japanese in _TRACK_NAMES.items():
            if japanese == name or code.name.lower() == name.lower():
                return code
        raise ValueError(f"Unknown track name: {name}")


_TRACK_NAMES: dict["TrackCode", str] = {
    TrackCode.KAWAGUCHI: "川口",
    TrackCode.ISESAKI: "伊勢崎",
    TrackCode.HAMAMATSU: "浜松",
    TrackCode.IIZUKA: "飯塚",
    TrackCode.SANYO: "山陽",
}


class TrackCondition(str, Enum):
    """走路状態"""

    GOOD = "良"
    WET = "湿"
    HEAVY = "重"
    MIXED = "斑"


class RiderRank(str, Enum):
    """選手ランク"""

    S = "S"
    A = "A"
    B = "B"


class EntryStatus(str, Enum):
    """出走状態"""

    RACING = "出走"
    CANCELLED = "取消"
    FELL = "落車"
    DISQUALIFIED = "失格"


class RaceStatus(str, Enum):
    """レース成立状態"""

    NORMAL = "正常"
    PARTIAL_CANCEL = "一部取消"
    VOID = "不成立"


class TicketType(str, Enum):
    """車券種別"""

    WIN = "単勝"
    PLACE = "複勝"
    EXACTA = "2連単"
    QUINELLA = "2連複"
    WIDE = "ワイド"
    TRIFECTA = "3連単"
    TRIO = "3連複"


class Grade(str, Enum):
    """レースグレード"""

    SG = "SG"
    GI = "GI"
    GII = "GII"
    NORMAL = "普通"
