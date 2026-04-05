# データソースと仕様リファレンス

## 1. データソース一覧

| ソース | 取得データ | 方式 | 費用 |
|--------|-----------|------|------|
| **autorace.jp** | 出走表、試走タイム、レース結果、払い戻し | HTTP（JSON + HTML） | 無料 |
| **oddspark.com** | 市場オッズ、発走時刻 | HTTP（HTML スクレイピング） | 無料 |

---

## 2. autorace.jp

### 2.1 認証

- **CSRF トークン**が必須。API リクエスト前に CSRF を取得する必要がある
- 実装: `src/providence/scraper/autorace_jp.py`

### 2.2 取得可能データ

| データ | API / ページ | 更新タイミング |
|--------|-------------|---------------|
| 開催情報 | `situationList` | 当日 |
| 出走表（Program） | Race JSON | レース前 |
| 試走タイム | Program 内 | レース前（試走後） |
| 選手プロフィール | Player JSON | 随時 |
| レース結果（着順） | Result JSON | レース後 |
| 払い戻し | Result HTML | レース後 |

### 2.3 レース状態の判定

`race.js` の `situationList` から取得:
- **`nowRaceNo`**: 現在発売中 or 次発走のレース番号
- **`On Sale Race`**: 発売中レース
- **`Last Result`**: 直前の確定結果

### 2.4 払い戻しキー対応

autorace.jp の払い戻し HTML で使用されるキーと `TicketType` の対応:

| HTML キー | TicketType | 券種 |
|----------|-----------|------|
| `tns` | WIN | 単勝 |
| `fns` | PLACE | 複勝 |
| `rtw` | EXACTA | 2 連単 |
| `rfu` | QUINELLA | 2 連複 |
| `wid` | WIDE | ワイド |
| `srt` | TRIFECTA | 3 連単 |
| `sfu` | TRIO | 3 連複 |

### 2.5 走路状態コード

autorace.jp が返す走路状態の文字列:
- `良`, `湿`, `斑`, `重` → `TrackCondition` に変換
- `風`, `油`, `荒` 等は走路状態としては `None`（非標準値）

---

## 3. oddspark.com

### 3.1 概要

- パス: `/autorace/`
- 場コード: 02-06（autorace.jp と同じ）
- 主な用途: 市場オッズの取得、発走時刻の取得

### 3.2 取得可能データ

| データ | 用途 |
|--------|------|
| 市場オッズ | 全券種のオッズと人気順 |
| 発走時刻 | `Race.scheduled_start_at`, `Race.telvote_close_at` の更新 |

### 3.3 オッズの時系列管理

- オッズは `odds_snapshot` テーブルに `ingestion_batch_id` 付きで格納
- 同一バッチ内のオッズは同時点のスナップショットとして扱う
- `judgment_time` 以前の最新バッチのみを予測・バックテストで使用

---

## 4. データスキーマ（Pydantic）

`src/providence/scraper/schemas.py` で定義。スクレイパーが正規化した後のデータ構造。

### 4.1 主要スキーマ

| スキーマ | 内容 | 主なフィールド |
|---------|------|---------------|
| `RaceEntriesResponse` | 出走表 | 場、日付、レース番号、グレード、距離（既定 3100）、天候、走路、気温、湿度、走路温度 |
| `EntryRow` | 出走車 1 台 | 枠番、ハンデ、試走タイム、平均試走、偏差、レーススコア |
| `RefundRow` | 払い戻し 1 行 | 券種、組み合わせ、払戻金額 |
| `OddsRow` / `OddsResponse` | オッズ | 券種、組み合わせ、オッズ値、人気 |
| `PlayerProfileResponse` | 選手情報 | 登録番号、名前、年齢、世代、ランク、所属場 |

### 4.2 バリデーション制約

- `post_position`: 1-8
- `race_number`: 1-12
- `handicap_meters`: 0-110

---

## 5. DB スキーマ概要

`src/providence/database/tables.py` に定義。

| テーブル | 概要 | ユニーク制約 |
|---------|------|-------------|
| `tracks` | 場マスタ | `id` |
| `riders` | 選手マスタ | `registration_number` |
| `races` | レース情報 | `track_id + race_date + race_number` |
| `race_entries` | 出走車 | `race_id + post_position` |
| `race_results` | レース結果 | `race_entry_id` |
| `odds_snapshot` | 市場オッズ | `race_id + ticket_type + combination + ingestion_batch_id` |
| `ticket_payouts` | 払い戻し | `race_id + ticket_type + combination` |
| `strategy_runs` | 戦略実行 | — |
| `predictions` | 予測結果 | — |
| `betting_log` | 購入記録 | — |
| `simulation_runs` | バックテスト | `semantic_key` |
| `model_performance` | モデル性能 | `model_version + race_date + window_label` |
| `feedback_runs` | フィードバック | — |
| `scrape_log` | スクレイプ履歴 | — |

### 5.1 races テーブルの主要カラム

グレード、距離（既定 3100）、走路状態、天候、気温、湿度、走路温度、レース成立状態、発走予定時刻（`scheduled_start_at`）、電話投票締切時刻（`telvote_close_at`）。

### 5.2 race_entries テーブルの主要カラム

枠番（`post_position`）、ハンデ（`handicap_meters`）、試走タイム（`trial_time`）、平均試走タイム、試走偏差、レーススコア、出走状態（`entry_status`）。

### 5.3 オッズと払い戻しの分離

Phase 3 で分離された設計:
- **`odds_snapshot`**: 市場オッズ専用。`ingestion_batch_id` で時系列管理
- **`ticket_payouts`**: 確定払い戻し専用。精算・バックテストで使用
