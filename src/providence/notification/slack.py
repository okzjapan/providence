"""Slack notification for autorace prediction results with rank classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import httpx
import structlog

from providence.services.prediction_runner import PredictionResult
from providence.strategy.normalize import format_combination

log = structlog.get_logger()

SLACK_MENTION_OKZ = "<@U070Q54JAAC>"


class BetRank(StrEnum):
    """投票ランク。predict-next-race.md のランク定義に基づく。"""

    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


@dataclass(frozen=True)
class RankJudgment:
    rank: BetRank
    label: str
    is_recommended: bool


_RANK_LABELS: dict[BetRank, tuple[str, bool]] = {
    BetRank.S: ("★投票推奨", True),
    BetRank.A: ("投票可", False),
    BetRank.B: ("慎重に", False),
    BetRank.C: ("非推奨", False),
    BetRank.D: ("回避", False),
    BetRank.E: ("投票不可", False),
}


def judge_rank(result: PredictionResult) -> RankJudgment:
    """PredictionResult からランクを判定する。

    ランク定義（predict-next-race.md より）:
    - S: Confidence >= 0.90 AND 最良 EV >= 0.40 → バックテスト ROI +54.5%
    - A: Confidence >= 0.90 AND 最良 EV >= 0.20
    - B: Confidence >= 0.80 AND 最良 EV >= 0.10
    - C: Confidence >= 0.50 AND 最良 EV >= 0.0
    - D: Confidence < 0.50 or EV < 0
    - E: skip_reason あり（候補なし）
    """
    strategy = result.strategy

    if strategy.skip_reason:
        if strategy.skip_reason == "no_market_odds":
            rank = BetRank.E
        elif strategy.skip_reason == "low_confidence":
            rank = BetRank.D
        elif strategy.skip_reason == "rounded_below_minimum":
            rank = BetRank.C
        else:
            rank = BetRank.D
        label, recommended = _RANK_LABELS[rank]
        return RankJudgment(rank=rank, label=label, is_recommended=recommended)

    conf = strategy.confidence_score
    bets = strategy.recommended_bets or strategy.candidate_bets
    best_ev = max((b.expected_value for b in bets), default=-1.0)

    if conf >= 0.90 and best_ev >= 0.40:
        rank = BetRank.S
    elif conf >= 0.90 and best_ev >= 0.20:
        rank = BetRank.A
    elif conf >= 0.80 and best_ev >= 0.10:
        rank = BetRank.B
    elif conf >= 0.50 and best_ev >= 0.0:
        rank = BetRank.C
    else:
        rank = BetRank.D

    label, recommended = _RANK_LABELS[rank]
    return RankJudgment(rank=rank, label=label, is_recommended=recommended)


def send_prediction_to_slack(
    webhook_url: str,
    result: PredictionResult,
    *,
    mention: str = "",
) -> bool:
    """予測結果を Slack に通知する。

    全レースの結果を通知し、ランク判定を付与する。
    S ランク（投票推奨）の場合のみ mention を付与する。
    """
    payload = build_slack_message(result, mention=mention)
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10.0)
        resp.raise_for_status()
        judgment = judge_rank(result)
        log.info(
            "slack_notification_sent",
            track=result.track_code.name,
            race=result.race_number,
            rank=judgment.rank.value,
            is_recommended=judgment.is_recommended,
        )
        return True
    except httpx.HTTPError as exc:
        log.error(
            "slack_notification_failed",
            track=result.track_code.name,
            race=result.race_number,
            error=str(exc),
        )
        return False


def build_slack_message(
    result: PredictionResult,
    *,
    mention: str = "",
) -> dict:
    """PredictionResult から Slack Block Kit メッセージを構築する。"""
    track = result.track_code.japanese_name
    header = f"{track} {result.target_date} R{result.race_number}"
    judgment = judge_rank(result)

    if result.strategy.skip_reason:
        return _build_skip_message(header, result, judgment)

    if result.strategy.recommended_bets:
        return _build_recommendation_message(header, result, judgment, mention=mention)

    return _build_no_bet_message(header, result, judgment)


def _build_skip_message(header: str, result: PredictionResult, judgment: RankJudgment) -> dict:
    conf = result.strategy.confidence_score
    reason = result.strategy.skip_reason
    text = f"ランク *{judgment.rank}*（{judgment.label}）| ⏭ {header} | 信頼度 {conf:.2f} | {reason}"

    win_section = _format_win_probs(result)
    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*勝率*\n{win_section}"}},
    ]
    return {"blocks": blocks, "text": text}


def _build_recommendation_message(
    header: str,
    result: PredictionResult,
    judgment: RankJudgment,
    *,
    mention: str = "",
) -> dict:
    strategy = result.strategy
    total = strategy.total_recommended_bet
    conf = strategy.confidence_score

    win_section = _format_win_probs(result)
    bet_section = _format_recommended_bets(strategy.recommended_bets)

    rank_display = f"ランク *{judgment.rank}*（{judgment.label}）"
    mention_prefix = f"{mention} " if mention and judgment.is_recommended else ""
    summary = f"{mention_prefix}{rank_display} | 🏁 *{header}* | 信頼度 {conf:.2f} | 合計推奨 *¥{total:,.0f}*"

    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*勝率*\n{win_section}"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*推奨買い目*\n{bet_section}"}},
    ]

    if not judgment.is_recommended:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": "⚠ S ランク基準（信頼度>=0.90, EV>=0.40）未達のため参考表示"},
        ]})

    if not result.has_market_odds:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": "⚠ オッズ未取得のため推奨精度が低い可能性"},
        ]})

    if result.missing_trial_positions:
        cars = ", ".join(str(p) for p in result.missing_trial_positions)
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"⚠ 試走タイム未取得: {cars}号車"},
        ]})

    return {
        "blocks": blocks,
        "text": f"{'★' if judgment.is_recommended else ''}{judgment.rank} {header} ¥{total:,.0f}",
    }


def _build_no_bet_message(header: str, result: PredictionResult, judgment: RankJudgment) -> dict:
    conf = result.strategy.confidence_score
    win_section = _format_win_probs(result)

    text = f"ランク *{judgment.rank}*（{judgment.label}）| 📊 {header} | 信頼度 {conf:.2f} | 推奨なし"
    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*勝率*\n{win_section}"}},
    ]

    if not result.has_market_odds:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": "⚠ オッズ未取得"},
        ]})

    return {
        "blocks": blocks,
        "text": text,
    }


def _format_win_probs(result: PredictionResult) -> str:
    bundle = result.bundle
    lines = []
    for index, prob in sorted(bundle.ticket_probs["win"].items(), key=lambda x: x[1], reverse=True):
        car = bundle.index_map.post_position_for_index(index)
        bar = "█" * int(prob * 20)
        lines.append(f"`{car}` {bar} {prob:.1%}")
    return "\n".join(lines)


def _format_recommended_bets(bets: list) -> str:
    lines = []
    for bet in sorted(bets, key=lambda b: b.recommended_bet, reverse=True):
        combo = format_combination(bet.ticket_type, bet.combination)
        lines.append(
            f"`{bet.ticket_type.value}` {combo}  "
            f"確率 {bet.probability:.1%}  オッズ {bet.odds_value:.1f}  "
            f"EV {bet.expected_value:+.2f}  → *¥{bet.recommended_bet:,.0f}*"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Daily overview (batch Phase 1 summary)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RaceOverviewEntry:
    """1レース分のサマリー情報。"""

    track_name: str
    race_number: int
    scheduled_start: datetime | None
    confidence: float
    is_candidate: bool
    top3_win_probs: list[tuple[int, float]]


def send_daily_overview_to_slack(
    webhook_url: str,
    target_date: str,
    entries: list[RaceOverviewEntry],
    min_confidence: float,
) -> bool:
    """全レースの信頼度サマリーを1つの Slack メッセージで送信する。"""
    payload = _build_daily_overview_message(target_date, entries, min_confidence)
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10.0)
        resp.raise_for_status()
        log.info("daily_overview_sent", races=len(entries))
        return True
    except httpx.HTTPError as exc:
        log.error("daily_overview_failed", error=str(exc))
        return False


def _build_daily_overview_message(
    target_date: str,
    entries: list[RaceOverviewEntry],
    min_confidence: float,
) -> dict:
    if not entries:
        return {"text": f"📋 {target_date}: 開催なし", "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"📋 {target_date}: 開催なし"}},
        ]}

    tracks = sorted({e.track_name for e in entries})
    candidates = [e for e in entries if e.is_candidate]
    non_candidates = [e for e in entries if not e.is_candidate]

    header = f"📋 *{target_date} 予測サマリー*（{', '.join(tracks)} 全{len(entries)}R）"

    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
    ]

    if candidates:
        lines = []
        for e in sorted(candidates, key=lambda x: (x.track_name, x.race_number)):
            start_str = e.scheduled_start.strftime("%H:%M") if e.scheduled_start else "??:??"
            top3 = "  ".join(f"{car}号車{p:.0%}" for car, p in e.top3_win_probs[:3])
            lines.append(f"✅ {e.track_name} R{e.race_number} ({start_str}) 信頼度 *{e.confidence:.2f}*  {top3}")
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn",
            "text": f"*投票候補（信頼度 >= {min_confidence}）*\n" + "\n".join(lines),
        }})
    else:
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn", "text": f"*投票候補なし*（全レース信頼度 < {min_confidence}）",
        }})

    if non_candidates:
        blocks.append({"type": "divider"})
        lines = []
        for e in sorted(non_candidates, key=lambda x: (x.track_name, x.race_number)):
            start_str = e.scheduled_start.strftime("%H:%M") if e.scheduled_start else "??:??"
            lines.append(f"❌ {e.track_name} R{e.race_number} ({start_str}) 信頼度 {e.confidence:.2f}")
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn", "text": "*非推奨*\n" + "\n".join(lines),
        }})

    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": "※ 試走タイム未取得のため信頼度は暫定値。投票候補は発走前に最終判定します"},
    ]})

    n_cand = len(candidates)
    text = f"📋 {target_date} {len(entries)}R中 {n_cand}R が投票候補"
    return {"blocks": blocks, "text": text}
