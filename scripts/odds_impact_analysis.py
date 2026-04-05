"""オートレース単勝: 購入額がオッズに与える影響の簡易分析.

パリミュチュエル方式の計算式と、公開売上データから推定した
投票プールサイズを組み合わせて、購入額ごとのオッズ変動を算出する。

Usage:
    uv run python scripts/odds_impact_analysis.py
"""

from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "providence.db"
TAKEOUT = 0.30  # 2012年改正後の控除率


# ------------------------------------------------------------------ #
#  1. DB から単勝オッズの実態を把握
# ------------------------------------------------------------------ #

def fetch_odds_stats(db: str) -> dict:
    """直近データから単勝オッズの統計を返す."""
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(DISTINCT os.race_id)
        FROM odds_snapshot os
        WHERE os.ticket_type = '単勝'
    """)
    total_races = cur.fetchone()[0]

    # 各レースの最新バッチの単勝オッズ全選手を取得
    # captured_at が最大のスナップショット群を1レースにつき1セットだけ取る
    cur.execute("""
        WITH ranked AS (
            SELECT os.race_id, os.combination, os.odds_value,
                   os.ingestion_batch_id, os.captured_at,
                   r.race_date, t.name as track, r.race_number, r.grade,
                   ROW_NUMBER() OVER (
                       PARTITION BY os.race_id, os.combination
                       ORDER BY os.captured_at DESC, os.id DESC
                   ) as rn
            FROM odds_snapshot os
            JOIN races r ON r.id = os.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE os.ticket_type = '単勝'
        )
        SELECT race_id, race_date, track, race_number, grade,
               combination, odds_value
        FROM ranked
        WHERE rn = 1
        ORDER BY race_id, CAST(combination AS INTEGER)
    """)
    rows = cur.fetchall()
    conn.close()

    races: dict[int, dict] = {}
    for race_id, race_date, track, race_no, grade, combo, odds in rows:
        if race_id not in races:
            races[race_id] = {
                "date": race_date, "track": track, "race_no": race_no,
                "grade": grade, "odds": [],
            }
        races[race_id]["odds"].append(odds)

    # フルオッズ (6選手以上) のレースのみ統計対象
    all_odds: list[float] = []
    inverse_sums: list[float] = []
    runner_counts: list[int] = []
    min_odds_list: list[float] = []
    max_odds_list: list[float] = []
    partial_count = 0

    for r in races.values():
        odds_list = r["odds"]
        if not odds_list:
            continue
        if len(odds_list) < 6:
            partial_count += 1
            continue
        all_odds.extend(odds_list)
        inverse_sums.append(sum(1.0 / o for o in odds_list))
        runner_counts.append(len(odds_list))
        min_odds_list.append(min(odds_list))
        max_odds_list.append(max(odds_list))

    return {
        "total_races": total_races,
        "analyzed_races": len(races) - partial_count,
        "partial_count": partial_count,
        "avg_runners": statistics.mean(runner_counts) if runner_counts else 0,
        "avg_inverse_sum": statistics.mean(inverse_sums) if inverse_sums else 0,
        "median_odds": statistics.median(all_odds) if all_odds else 0,
        "mean_odds": statistics.mean(all_odds) if all_odds else 0,
        "odds_p10": sorted(all_odds)[len(all_odds) // 10] if all_odds else 0,
        "odds_p90": sorted(all_odds)[len(all_odds) * 9 // 10] if all_odds else 0,
        "avg_min_odds": statistics.mean(min_odds_list) if min_odds_list else 0,
        "avg_max_odds": statistics.mean(max_odds_list) if max_odds_list else 0,
        "median_min_odds": statistics.median(min_odds_list) if min_odds_list else 0,
        "median_max_odds": statistics.median(max_odds_list) if max_odds_list else 0,
    }


# ------------------------------------------------------------------ #
#  2. パリミュチュエル計算
# ------------------------------------------------------------------ #

def calc_new_odds(pool: float, current_odds: float, bet: float, takeout: float = TAKEOUT) -> float:
    """自分が bet 円追加購入した後の新オッズ（0.1刻み切捨て前）."""
    payout_rate = 1.0 - takeout
    current_bet_on_runner = pool * payout_rate / current_odds
    new_pool = pool + bet
    new_bet_on_runner = current_bet_on_runner + bet
    return new_pool * payout_rate / new_bet_on_runner


def calc_new_odds_rounded(pool: float, current_odds: float, bet: float, takeout: float = TAKEOUT) -> float:
    """切り捨て後の新オッズ（0.1刻み）."""
    raw = calc_new_odds(pool, current_odds, bet, takeout)
    return int(raw * 10) / 10.0


def find_threshold_bet(pool: float, current_odds: float, target_change: float, takeout: float = TAKEOUT) -> int:
    """オッズを target_change だけ下げるのに必要な最小購入額（100円単位）."""
    for bet in range(100, 10_000_100, 100):
        new = calc_new_odds(pool, current_odds, bet, takeout)
        if current_odds - new >= target_change:
            return bet
    return -1


# ------------------------------------------------------------------ #
#  3. プール推定
# ------------------------------------------------------------------ #

DAILY_SALES = {
    "普通": 136_525_528,
    "ミッドナイト等": 123_142_045,
    "G2": 189_812_967,
    "G1": 244_999_562,
    "SG": 379_909_266,
}
RACES_PER_DAY = 12
WIN_SHARE_LOW = 0.03
WIN_SHARE_MID = 0.04
WIN_SHARE_HIGH = 0.06


def estimate_pools() -> dict[str, dict]:
    """グレード × 単勝シェアから投票プールを推定."""
    result = {}
    for grade, daily in DAILY_SALES.items():
        per_race = daily / RACES_PER_DAY
        result[grade] = {
            "daily": daily,
            "per_race_total": per_race,
            "win_pool_low": per_race * WIN_SHARE_LOW,
            "win_pool_mid": per_race * WIN_SHARE_MID,
            "win_pool_high": per_race * WIN_SHARE_HIGH,
        }
    return result


# ------------------------------------------------------------------ #
#  4. メイン出力
# ------------------------------------------------------------------ #

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    # --- DB統計 ---
    print_section("1. DBデータからの単勝オッズ統計")
    if DB_PATH.exists():
        stats = fetch_odds_stats(str(DB_PATH))
        print(f"  DB内全レース数     : {stats['total_races']:,}")
        print(f"  フルオッズ有レース : {stats['analyzed_races']:,}")
        print(f"  (払戻のみレース    : {stats['partial_count']:,})")
        print(f"  平均出走数         : {stats['avg_runners']:.1f}")
        print(f"  Σ(1/odds) 平均    : {stats['avg_inverse_sum']:.3f}")
        print(f"    → 理論値 1/(1-0.30) = {1/(1-TAKEOUT):.3f}")
        print(f"    → 差分 (端数利益)   : {stats['avg_inverse_sum'] - 1/(1-TAKEOUT):.3f}")
        print(f"  オッズ中央値       : {stats['median_odds']:.1f}")
        print(f"  オッズ平均値       : {stats['mean_odds']:.1f}")
        print(f"  10パーセンタイル   : {stats['odds_p10']:.1f}")
        print(f"  90パーセンタイル   : {stats['odds_p90']:.1f}")
        print(f"  1番人気 平均オッズ : {stats['avg_min_odds']:.1f}")
        print(f"  最低人気 平均オッズ : {stats['avg_max_odds']:.1f}")
        print(f"  1番人気 中央オッズ : {stats['median_min_odds']:.1f}")
        print(f"  最低人気 中央オッズ : {stats['median_max_odds']:.1f}")
    else:
        print("  DB が見つかりません。公開データのみで分析します。")

    # --- プール推定 ---
    print_section("2. 単勝投票プールの推定 (FY2024 実績ベース)")
    print(f"  控除率: {TAKEOUT*100:.0f}%  |  単勝シェア推定: {WIN_SHARE_LOW*100:.0f}-{WIN_SHARE_HIGH*100:.0f}%")
    print()
    pools = estimate_pools()
    print(f"  {'グレード':<14} {'1日売上':>14} {'1R総売上':>12} {'単勝プール(推定)':>20}")
    print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*20}")
    for grade, p in pools.items():
        low = p["win_pool_low"]
        high = p["win_pool_high"]
        mid = p["win_pool_mid"]
        print(f"  {grade:<14} {p['daily']/1e6:>11.1f}万 {p['per_race_total']/1e6:>9.1f}万 "
              f"{low/1e4:>6.0f}〜{high/1e4:>5.0f}万 (中央{mid/1e4:>.0f}万)")

    print()
    print("  ※ 単勝シェアは競馬(JRA)の約5.5%を参考に、オートレースでは")
    print("    やや低めの 3-6% と推定。レース番号による偏りは考慮していない。")
    print("    (メインレースは平均の 2-3倍、序盤レースは 1/2-1/3 の可能性あり)")

    # --- 影響度テーブル ---
    print_section("3. 購入額がオッズに与える影響")

    bet_amounts = [100, 500, 1000, 2000, 3000, 5000, 10000]
    odds_scenarios = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    pool_scenarios = [
        ("ミッドナイト序盤", 150_000),
        ("普通レース前半", 300_000),
        ("普通レース平均", 450_000),
        ("普通レース後半", 700_000),
        ("G1/SG メイン", 1_500_000),
    ]

    for pool_name, pool in pool_scenarios:
        print(f"\n  ■ プール: {pool_name} ({pool/1e4:.0f}万円)")
        print(f"    {'購入額':>8} | ", end="")
        for o in odds_scenarios:
            print(f"  {o:>5.1f}倍", end="")
        print()
        print(f"    {'-'*8}-+-{'-' * (8 * len(odds_scenarios))}")

        for bet in bet_amounts:
            print(f"    {bet:>6,}円 | ", end="")
            for odds in odds_scenarios:
                new = calc_new_odds(pool, odds, bet)
                change = new - odds
                pct = change / odds * 100
                print(f"  {pct:>+5.1f}%", end="")
            print()

    # --- 実用的な閾値テーブル ---
    print_section("4. オッズが 0.1 下がる最小購入額 (100円単位)")
    print(f"\n  オッズが表示上 0.1 下がるために必要な金額の目安:")
    print(f"    {'プール':>18} | ", end="")
    for o in [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]:
        print(f" {o:>5.1f}倍", end="")
    print()
    print(f"    {'-'*18}-+-{'-' * (7 * 6)}")

    for pool_name, pool in pool_scenarios:
        print(f"    {pool_name:>18} | ", end="")
        for o in [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]:
            threshold = find_threshold_bet(pool, o, 0.1)
            if threshold > 0:
                if threshold >= 10000:
                    print(f" {threshold/1e4:>4.1f}万", end="")
                else:
                    print(f" {threshold:>5,}円", end="")
            else:
                print(f"   >1千万", end="")
        print()

    # --- プールに対する割合 ---
    print_section("5. 購入額がプールに占める割合")
    print(f"\n  自分の購入額がプール全体の何%を占めるか (= 影響度の目安):")
    print(f"    {'購入額':>8} | ", end="")
    for _, pool in pool_scenarios:
        print(f" {pool/1e4:>5.0f}万", end="")
    print()
    print(f"    {'-'*8}-+-{'-' * (7 * len(pool_scenarios))}")

    for bet in bet_amounts:
        print(f"    {bet:>6,}円 | ", end="")
        for _, pool in pool_scenarios:
            pct = bet / pool * 100
            print(f" {pct:>5.1f}%", end="")
        print()

    # --- 結論 ---
    print_section("6. まとめ・実用ガイドライン")
    print("""
  ■ オートレースの単勝プールは推定 15万〜150万円/レース と非常に小さい。
    (競馬 JRA の単勝プール: 数億〜数十億円/レースとは桁違い)

  ■ オッズへの影響度の目安:

    ┌────────────────────────────────────────────────────────┐
    │  購入額      影響度          プールに対する割合         │
    │  ──────────────────────────────────────────────────── │
    │  100〜500円   ほぼ影響なし     0.01〜0.3%              │
    │  1,000円      軽微             0.1〜0.7%               │
    │  3,000円      やや影響あり     0.2〜2.0%               │
    │  5,000円      注意が必要       0.3〜3.3%               │
    │  10,000円     明確に影響       0.7〜6.7%               │
    │  30,000円〜   大きく影響       2〜20%+                  │
    └────────────────────────────────────────────────────────┘

  ■ 実運用の目安:
    - 1レース 1,000円以下: オッズ影響を気にする必要なし
    - 1レース 3,000円前後: 人気薄(高オッズ)では影響が出始める
    - 1レース 5,000〜10,000円: 低人気選手のオッズが目に見えて動く
    - 1レース 10,000円超: 全般的にオッズ変動を前提とした戦略が必要

  ■ 特に注意すべきケース:
    - ミッドナイトレース序盤: プールが極端に小さい (推定15万〜)
    - 低人気選手への単勝: プールシェアが小さいため影響が増幅される
    - 締切間際の購入: オッズ反映が遅れ、想定と異なるオッズで購入になるリスク

  ■ 前提条件・限界:
    - 単勝プールの絶対額は公開されていないため、公開売上データから推定
    - 単勝の売上シェア 3-6% は競馬のデータから類推（実際は異なる可能性あり）
    - レース番号やグレードによるプール偏差は考慮していない
    - ネット投票比率 ~75% による締切直前集中の影響は未考慮
""")


if __name__ == "__main__":
    main()
