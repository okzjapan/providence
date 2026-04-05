---
name: autorace-domain
description: >-
  オートレース予想システムのドメイン制約とコンテキスト読み込み。
  オートレース関連のコード作成・レビュー・特徴量設計・DB設計・テスト作成時に自動適用。
  features/, scraper/, domain/enums.py, database/tables.py, autorace/ の変更時に適用。
  オートレース、選手、試走、車券、走路に言及するタスクで自動適用。
---

# オートレースドメインスキル

## Step 1: コンテキストの読み込み（最初に必ず実行）

オートレース関連のコードを変更する前に、以下のドキュメントとコードを読み込んでコンテキストを取得すること。

### 必ず読むドキュメント

タスクの種類に応じて、以下のドキュメントを Read ツールで読み込む:

| タスクの種類 | 読むべきドキュメント |
|-------------|-------------------|
| 全般（最低限） | `docs/autorace/glossary.md`（命名規則） |
| 特徴量の追加・変更 | `docs/autorace/02-feature-catalog.md` |
| Enum・ドメインモデル変更 | `docs/autorace/01-autorace-structure.md` |
| スクレイパー・DB変更 | `docs/autorace/03-data-sources.md` |
| モデル設計変更 | `docs/autorace/04-model-design.md` |
| 戦略パラメータ変更 | `docs/autorace/05-strategy-and-market.md` |
| コードレビュー | `docs/autorace/06-pitfalls.md` |

### 必ず読むソースコード

変更対象に応じて、以下の関連コードを読み込む:

| 変更対象 | 読むべきコード |
|---------|---------------|
| 特徴量 | `src/providence/features/pipeline.py`（パイプライン順序・除外列・カテゴリマップ） |
| 特徴量（試走系） | `src/providence/features/trial_run.py` |
| 特徴量（選手系） | `src/providence/features/rider.py` |
| 特徴量（レース内） | `src/providence/features/race.py` |
| 特徴量（場・環境） | `src/providence/features/track.py` |
| Enum | `src/providence/domain/enums.py` |
| スクレイパー | `src/providence/scraper/autorace_jp.py` または `oddspark.py` |
| DB スキーマ | `src/providence/database/tables.py` |
| 確率モデル | `src/providence/probability/plackett_luce.py` |
| 戦略 | `src/providence/strategy/types.py`（StrategyConfig）, `optimizer.py` |

## Step 2: ドメイン制約を適用してコードを書く

### 競技の基本制約

- **出走数**: 最大 8 車（枠番 1-8）
- **距離**: 原則 3,100m（6 周）
- **レース場**: 5 場（川口=2, 伊勢崎=3, 浜松=4, 飯塚=5, 山陽=6）
- **控除率**: 全車券一律 **25%**

### リーク防止ルール

予測時点で使えない情報を特徴量に含めてはならない。

**使用禁止**:
- `finish_position`（ラベルとしてのみ使用）
- `race_time`, `start_timing`, `accident_code`（レース後に確定）
- 落車・失格の `entry_status`（レース後に確定。出走取消は予測前に判明）

**特徴量計算時の原則**:
- 選手成績の集計は `d < current_date` の条件で過去データのみ使用
- `load_history(as_of_date)` で同日リークを防止
- `assert_no_leakage` テストを必ず実装

### 特徴量パイプライン実行順序

```
_prepare_base → add_trial_run_features → add_rider_features
  → add_race_features → add_track_features → _encode_categoricals
  → assert_no_leakage
```

`add_race_features` は `add_rider_features` に依存。順序を変えてはならない。

### ラベル設計

ランキング学習（LambdaRank）用: 1着=3, 2着=2, 3着=1, 4着以下=0

### カテゴリエンコーディング

```python
TRACK_CONDITION_MAP = {"良": 0, "湿": 1, "重": 2, "斑": 3, "__NULL__": 4}
WEATHER_MAP = {"晴": 0, "曇": 1, "雨": 2, "小雨": 3, "小雪": 4, "雪": 5, "other": 6, "__NULL__": 7}
GRADE_MAP = {"普通": 0, "GII": 1, "GI": 2, "SG": 3, "__NULL__": 4}
```

### 既知の制限

- 複勝（`TicketType.PLACE`）は Kelly 最適化の `_ticket_hits` に未対応
- 場・走路別の適性は最低 5 走のサンプルが必要（不足時は None）

## Step 3: ドキュメントを同時に更新

コード変更後、`.claude/skills/domain-doc-sync/SKILL.md` のマッピングに従い、対応するドキュメントを更新する。

## 詳細リファレンス

- 競技構造: `docs/autorace/01-autorace-structure.md`
- 特徴量カタログ: `docs/autorace/02-feature-catalog.md`
- データソース: `docs/autorace/03-data-sources.md`
- モデル設計: `docs/autorace/04-model-design.md`
- 車券戦略: `docs/autorace/05-strategy-and-market.md`
- 罠と対策: `docs/autorace/06-pitfalls.md`
- 用語集: `docs/autorace/glossary.md`
