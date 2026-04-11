# 特徴量カタログ

オートレース予想モデルで使用する特徴量の網羅的カタログ。
新しい特徴量を追加したら、このドキュメントを必ず更新すること。

実装: `src/providence/features/`

---

## カテゴリ一覧

| # | カテゴリ | 実装ファイル | 主な特徴量数 |
|---|---------|-------------|-------------|
| 1 | [基本情報](#1-基本情報) | `pipeline.py` | ~14 |
| 2 | [試走系](#2-試走系) | `trial_run.py` | ~14 |
| 3 | [選手成績系](#3-選手成績系) | `rider.py` | ~14 |
| 4 | [レース内文脈](#4-レース内文脈) | `race.py` | ~13 |
| 5 | [場・環境系](#5-場環境系) | `track.py` | ~3 |
| 6 | [前走詳細](#6-前走詳細) | `prev_race.py` | ~36 |
| 7 | [コンテキスト](#7-コンテキスト) | `context.py` | ~17 |

---

## パイプライン実行順序

```
_prepare_base (ベース整形・カテゴリマップ)
  → add_trial_run_features
  → add_rider_features
  → add_race_features     ← rider features に依存
  → add_track_features
  → _encode_categoricals
  → assert_no_leakage
```

---

## 1. 基本情報

`pipeline.py` の `_prepare_base` と `_encode_categoricals` で処理。

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `post_position` | 枠番 | int | 1-8 |
| `handicap_meters` | ハンデ（m） | int | 0-110。大きいほど不利 |
| `track_id` | 場 ID | categorical | 2-6（川口〜山陽） |
| `track_condition` | 走路状態 | int | 良=0, 湿=1, 重=2, 斑=3 |
| `weather` | 天候 | int | 晴=0 〜 雪=5, other=6, NULL=7 |
| `grade` | グレード | int | 普通=0, GII=1, GI=2, SG=3 |
| `trial_time` | 試走タイム（秒） | float | レース直前の計測値 |
| `avg_trial_time` | 平均試走タイム（秒） | float | |
| `race_score` | レーススコア | float | |

### カテゴリエンコーディング

| 変数 | 方式 | マッピング |
|------|------|-----------|
| `track_condition` | 序数 | 良=0, 湿=1, 重=2, 斑=3, NULL=4 |
| `weather` | 序数 | 晴=0, 曇=1, 雨=2, 小雨=3, 小雪=4, 雪=5, other=6, NULL=7 |
| `grade` | 序数 | 普通=0, GII=1, GI=2, SG=3, NULL=4 |
| `track_id` | そのまま（ID 値） | |

天候は出現回数 20 未満の稀な値を `other` にまとめるロジックあり。

### 派生属性（`_prepare_base` で生成）

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `is_home_track` | ホームトラックかどうか | int (0/1) | `home_track_id == track_id` |
| `rider_age` | 選手の年齢 | int | `race_date.year - birth_year` |
| `race_number` | レース番号 | int | 1-12。12R=メインレース |

### 除外カラム（特徴量に使用しない）

以下は `excluded_feature_columns` で学習から除外:

`row_key`, `race_id`, `race_entry_id`, `rider_id`, `rider_registration_number`, `race_date`, `finish_position`, `race_time`, `start_timing`, `accident_code`, `entry_status`, `race_status`, `avg_trial_time`, `race_score`, `birth_year`, `home_track_id`

---

## 2. 試走系

`trial_run.py` で計算。試走タイムはオートレース予測の最も重要なデータの一つ。

### 2.1 レース内試走指標

同一レース内での相対評価。

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `trial_time_rank` | レース内試走順位 | `trial_time.rank("dense").over("race_id")` | int |
| `field_avg_trial` | レース内平均試走タイム | `trial_time.mean().over("race_id")` | float |
| `field_trial_std` | レース内試走タイム標準偏差 | `trial_time.std().over("race_id")` | float |
| `trial_time_diff_from_best` | 最速試走との差 | `trial_time - min(trial_time)` (within race) | float |

### 2.2 選手別試走履歴

同一選手の過去試走タイムの推移。時系列安全（`d < current_date` で過去のみ使用）。

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `avg_trial_time_30d` | 直近 30 日間の平均試走タイム | 過去 30 日以内の試走平均 | float |
| `avg_trial_time_90d` | 直近 90 日間の平均試走タイム | 過去 90 日以内の試走平均 | float |
| `trial_time_trend` | 試走タイムトレンド | 直近 5 本の線形回帰傾き × (-1)。正=改善 | float |
| `trial_vs_avg_90d` | 今回試走 vs 90 日平均の差 | `trial_time - avg_trial_time_90d` | float |

**トレンド計算**: `np.polyfit(x, y, 1)[0]` の傾きを符号反転。値が大きいほど試走タイムが改善している（速くなっている）方向。

### 2.3 試走偏差系

過去の（競走タイム - 試走タイム）から算出する選手固有の「試走偏差」。良走路のみで計算。試走偏差が小さい選手は試走通りの走りをし、大きい選手は本番で遅くなりがち。

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `computed_trial_deviation` | 自前計算の試走偏差 | 過去の `race_time - trial_time` の直近10走平均（良走路のみ、最低5走必要） | float |
| `trial_deviation_trend` | 試走偏差トレンド | 直近 5 走の試走偏差の線形回帰傾き × (-1)。正=偏差縮小（改善） | float |
| `predicted_race_time` | 予想競走タイム | `trial_time + computed_trial_deviation` | float |
| `predicted_race_time_rank` | 予想競走タイム順位 | レース内での `predicted_race_time` の dense ランク | int |
| `trial_time_zscore` | 試走タイム Zスコア | `(trial_time - field_avg_trial) / field_trial_std`。std=0 時は 0.0 | float |

**リーク防止**: `race_time` は当該レースでは使用不可（`excluded_feature_columns`）だが、**過去レースの `race_time`** は `d < current_date` の条件で安全に利用可能。`computed_trial_deviation` は過去データのみから計算される。

**良走路限定の理由**: 湿・重走路ではタイヤグリップ特性が根本的に変わり、試走偏差の意味が異なる。良走路データのみを使うことで安定したベースライン偏差を提供する。

---

## 3. 選手成績系

`rider.py` で計算。過去の確定着順のみを使用（`position >= 1` のレースのみ履歴に追加）。時系列安全。

### 3.1 勝率・着率

| 変数名 | 定義 | ウィンドウ | 型 |
|--------|------|-----------|-----|
| `win_rate_10` | 直近 10 走の勝率 | 直近 10 走 | float |
| `top2_rate_10` | 直近 10 走の 2 着以内率 | 直近 10 走 | float |
| `top3_rate_10` | 直近 10 走の 3 着以内率 | 直近 10 走 | float |
| `win_rate_30` | 直近 30 走の勝率 | 直近 30 走 | float |
| `top3_rate_30` | 直近 30 走の 3 着以内率 | 直近 30 走 | float |

### 3.2 着順統計

| 変数名 | 定義 | ウィンドウ | 型 |
|--------|------|-----------|-----|
| `avg_finish_10` | 直近 10 走の平均着順 | 直近 10 走 | float |
| `avg_finish_30` | 直近 30 走の平均着順 | 直近 30 走 | float |
| `finish_std_10` | 直近 10 走の着順標準偏差 | 直近 10 走 | float |

### 3.3 トレンド・活動量

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `momentum` | 調子のトレンド | 直近 5 走の着順の線形回帰傾き × (-1)。正=改善 | float |
| `days_since_last_race` | 前走からの日数 | `current_date - last_race_date` | int |
| `total_races` | 通算レース数（予測時点まで） | — | int |

### 3.4 スタートタイミング履歴

過去レースのスタートタイミング（`race_results.start_timing`）を集計。オートレースの勝敗の 5 割以上はスタート展開で決まるとされる。時系列安全（過去のみ）。

| 変数名 | 定義 | ウィンドウ | 型 |
|--------|------|-----------|-----|
| `avg_start_timing_10` | 直近 10 走の平均スタートタイミング | 直近 10 走 | float |
| `avg_start_timing_30` | 直近 30 走の平均スタートタイミング | 直近 30 走 | float |
| `start_timing_consistency` | 直近 10 走のST標準偏差（安定性） | 直近 10 走 | float |

**リーク防止**: `start_timing` は当該レースでは `excluded_feature_columns`（レース後確定データ）。ここで使うのは **過去レースの start_timing** のみ。

---

## 4. レース内文脈

`race.py` で計算。add_rider_features の後に実行されるため、選手成績系の変数を利用可能。

### 4.1 ハンデ・試走の相対評価

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `handicap_rank` | レース内ハンデ順位 | `handicap_meters.rank("dense").over("race_id")` | int |
| `field_size` | 出走頭数 | `len().over("race_id")` | int |
| `handicap_vs_trial` | ハンデ順位と試走順位の差 | `handicap_rank - trial_time_rank` | int |
| `predicted_vs_handicap` | 予想タイム順位とハンデ順位の差 | `predicted_race_time_rank - handicap_rank` | int |

**`handicap_vs_trial` の解釈**: 正の値 = ハンデの割に試走が良い（実力以上の仕上がり）。負の値 = ハンデの割に試走が悪い。

**`predicted_vs_handicap` の解釈**: 正の値 = 予想競走タイムの割にハンデが軽い（有利）。負の値 = 予想タイムの割にハンデが重い。`handicap_vs_trial` の強化版で、試走偏差を考慮した実効タイムに基づく。

### 4.3 ハンデ構造・走路条件

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `handicap_range_in_field` | フィールド内ハンデ差 | `max(handicap_meters) - min(handicap_meters)` (within race) | int |
| `handicap_group` | ハンデ位置グループ | 0m=0, 10m=1, 20-30m=2, 40-50m=3, 60m+=4 | int |
| `is_wet_condition` | 湿・重走路フラグ | `track_condition` が「湿」or「重」なら 1 | int (0/1) |

**`handicap_group` の背景**: ハンデ位置によりスタートの叩きやすさが変わる。0-30m は大時計が近く攻めやすい。40-50m は最も叩きにくい。

### 4.2 メンバー強度

`win_rate_10` が利用可能な場合に計算される。

| 変数名 | 定義 | 計算式 | 型 |
|--------|------|--------|-----|
| `field_avg_win_rate` | レース内平均勝率 | `win_rate_10.mean().over("race_id")` | float |
| `field_max_win_rate` | レース内最高勝率 | `win_rate_10.max().over("race_id")` | float |
| `field_avg_finish` | レース内平均着順平均 | `avg_finish_10.mean().over("race_id")` | float |
| `form_rank_in_field` | レース内フォーム順位 | `win_rate_10.rank("dense", descending=True)` | int |
| `relative_win_strength` | 相対的勝力 | `win_rate_10 / field_avg_win_rate`（0-10 にクリップ） | float |
| `gap_to_top_rival` | トップライバルとの差 | `field_max_win_rate - win_rate_10` | float |

---

## 5. 場・環境系

`track.py` で計算。選手の場別・走路状態別の過去成績。最低 5 走のサンプルが必要。

| 変数名 | 定義 | 条件 | 型 |
|--------|------|------|-----|
| `rider_track_win_rate` | 同一場での勝率 | 同一 `track_id` で過去 5 走以上 | float |
| `rider_wet_win_rate` | 湿・重走路での勝率 | `track_condition` が「湿」or「重」で過去 5 走以上 | float |
| `rider_wet_top3_rate` | 湿・重走路での 3 着以内率 | 同上 | float |

---

## リーク防止

### 除外されるべき情報

以下は予測時点で利用不可、またはラベルそのものであるため特徴量に含めない:

| 情報 | 理由 |
|------|------|
| `finish_position` | ラベル（予測対象） |
| `race_time` | レース後に確定 |
| `start_timing` | レース後に確定 |
| `accident_code` | レース後に確定 |
| `entry_status`（取消・落車・失格） | 出走取消は予測前に判明するが、落車・失格はレース後 |

### リーク防止テスト

`pipeline.py` の `assert_no_leakage` で、特徴量 DataFrame にリーク候補カラムが含まれていないことを検証。

### 時系列安全性

全ての選手履歴特徴量は `d < current_date` の条件で過去データのみを使用。`load_history(as_of_date)` で同日リークも防止。

---

## 6. 前走詳細

`prev_race.py` で計算。各選手の直近5走の個別レースデータを抽出。時系列安全（`race_date < current_date` で過去のみ）。

### 6.1 前走個別データ

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `finish_position_prev1`〜`prev5` | 前1〜5走の着順 | int |
| `race_time_prev1`〜`prev5` | 前1〜5走の競走タイム | float |
| `start_timing_prev1`〜`prev5` | 前1〜5走のスタートタイミング | float |
| `handicap_prev1`〜`prev5` | 前1〜5走のハンデ | int |
| `track_id_prev1`〜`prev5` | 前1〜5走のレース場 | categorical |
| `is_wet_prev1`〜`prev5` | 前1〜5走が湿/重走路か | int (0/1) |

### 6.2 集約指標

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `form_score` | 直近5走の重み付き着順平均（0.5指数減衰、最新に高い重み） | float |
| `ewm_race_time` | 競走タイムの指数加重平均 | float |
| `ewm_start_timing` | STの指数加重平均 | float |
| `prev_best_finish` | 直近5走の最高着順 | int |
| `prev_worst_finish` | 直近5走の最低着順 | int |
| `prev_win_count` | 直近5走の勝利数 | int |
| `prev_top3_count` | 直近5走の3着以内数 | int |
| `prev_best_time` | 直近5走の最速タイム | float |
| `prev_time_std` | 直近5走のタイム標準偏差 | float |
| `finish_improvement` | 前走着順 vs その前の平均着順（正=改善） | float |

---

## 7. コンテキスト

`context.py` で計算。エンコーディング後に実行される。

### 7.1 時間・カレンダー

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `race_month` | レース月（1-12） | int |
| `rest_interval_group` | 休養日数グループ（0: <=3日, 1: <=7, 2: <=14, 3: <=30, 4: >30） | int |

### 7.2 ST場内相対

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `st_vs_field_avg` | 自分のST10走平均 - フィールド平均ST | float |
| `st_rank_in_field` | フィールド内ST順位 | int |

### 7.3 前走タイムZスコア

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `time_zscore_prev1`〜`prev5` | (前走タイム - EWMタイム) / タイム標準偏差 | float |

### 7.4 交互作用

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `temp_x_condition` | 気温 × 走路状態 | float |
| `humidity_x_condition` | 湿度 × 走路状態 | float |
| `home_x_track_wr` | ホームトラック × 場別勝率 | float |

---

## 欠損値の扱い

| ケース | 対象特徴量 | 処理 |
|--------|-----------|------|
| 新人選手・レース数不足 | `win_rate_10` 等の成績系 | None（LightGBM が自然に処理） |
| 同一場での出走 5 走未満 | `rider_track_win_rate` | None |
| 湿・重走路での出走 5 走未満 | `rider_wet_win_rate` | None |
| 試走未取得 | `trial_time` 系全て | None（CLI で警告表示） |
