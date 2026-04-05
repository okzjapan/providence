# 特徴量カタログ

競馬予想モデルで使用する特徴量の網羅的カタログ。
新しい特徴量を追加したら、このドキュメントを必ず更新すること。

先行研究では 170-250 次元の特徴量が使用されている。以下にカテゴリ別で全量を定義する。

---

## カテゴリ一覧

| # | カテゴリ | 次元数目安 | 実装 Phase |
|---|---------|-----------|-----------|
| 1 | [基本情報](#1-基本情報) | ~30 | Phase 2a |
| 2 | [パフォーマンス系](#2-パフォーマンス系) | ~50-60 | Phase 2a |
| 3 | [脚質・展開系](#3-脚質展開系) | ~30-40 | Phase 2b |
| 4 | [条件別適性](#4-条件別適性) | ~30 | Phase 2b |
| 5 | [血統適性](#5-血統適性) | ~20-30 | Phase 2b |
| 6 | [関係者成績](#6-関係者成績) | ~20 | Phase 2c |
| 7 | [調教データ](#7-調教データ) | ~15-20 | Phase 2c |
| 8 | [レース内相対評価](#8-レース内相対評価) | ~10 | Phase 2c |
| 9 | [馬体情報](#9-馬体情報) | ~5 | Phase 2a |
| 10 | [レーティング系](#10-レーティング系) | ~5 | Phase 5 |

---

## 特徴量の記述フォーマット

各特徴量は以下の形式で定義する:

- **名前**: コード上の変数名（[glossary.md](glossary.md) と一致させる）
- **定義**: 何を表す値か
- **計算式**: 具体的な計算方法（あれば）
- **型**: int / float / categorical
- **注意点**: リーク可能性、欠損パターン、前処理等

---

## 1. 基本情報

レースと出走馬の基本属性。予測時点で確定している情報のみ。

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `distance` | レース距離（m） | int | 1000-3600 |
| `post_position` | 馬番 | int | 1-18 |
| `draw` | 枠番 | int | 1-8 |
| `impost_weight` | 斤量（kg） | float | 馬齢/定量/別定/ハンデで決定方法が異なる |
| `age` | 馬齢 | int | 2-9+ |
| `sex` | 性別 | categorical | 牡/牝/セン |
| `num_runners` | 出走頭数 | int | 5-18 |
| `racecourse_id` | 競馬場 ID | categorical | 10 場 |
| `surface` | 芝 / ダート | categorical | モデル分離の基準 |
| `going` | 馬場状態 | categorical | 良/稍重/重/不良 |
| `course_type` | コース区分 | categorical | 内回り/外回り/直線 |
| `stable_location` | 所属 | categorical | 栗東/美浦 |
| `race_class` | クラス | categorical | 新馬〜G1（序数エンコーディング） |
| `race_day_number` | 開催日数 | int | トラックバイアスの代理変数 |
| `weight_rule` | 斤量制度 | categorical | 馬齢/定量/別定/ハンデ |
| `season` | 季節 | categorical | 春/夏/秋/冬 |
| `turf_type` | 芝質 | categorical | 野芝/洋芝 |
| `straight_length` | 直線距離（m） | float | 競馬場マスタから取得 |
| `elevation_diff` | 高低差（m） | float | 同上 |
| `direction` | 回り方向 | categorical | 左回り/右回り |
| `is_first_dirt` | 初ダートフラグ | bool | 芝→ダート転向検出 |
| `is_first_turf` | 初芝フラグ | bool | ダート→芝転向検出 |
| `distance_change` | 距離変更幅（m） | int | 前走との差。延長 +、短縮 - |
| `equipment_change` | 馬具変更フラグ | bool | ブリンカー新規着用等 |
| `blinkers` | ブリンカー着用 | bool | |
| `apprentice_allowance` | 減量幅（kg） | float | 0/1/2/3 |

---

## 2. パフォーマンス系

過去走の走破タイム・パフォーマンスに関する特徴量。各指標を過去 5 走分格納する。

### 2.1 スピード指数

西田式スピード指数の実装。

**計算式**:
```
speed_figure = (base_time - race_time) × distance_index + track_variant + (impost - 55) × 2 + 80
```

| 要素 | 定義 |
|------|------|
| `base_time` | 同コース古馬 500 万・1000 万条件の 1-3 着平均タイム |
| `distance_index` | `1 / (距離ごとの平均タイム) × 1000`。芝 2400m で約 6.8 |
| `track_variant` | 同日同馬場のタイム偏差平均（馬場指数） |
| `impost` | 斤量 |

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `prev{N}_speed_figure` | 過去 N 走のスピード指数 | float |
| `avg_speed_figure_5` | 過去 5 走のスピード指数平均 | float |
| `max_speed_figure_5` | 過去 5 走の最高スピード指数 | float |
| `std_speed_figure_5` | 過去 5 走の標準偏差（安定性） | float |

### 2.2 走破タイム

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `prev{N}_time_sec` | 過去 N 走の走破タイム（秒） | float | |
| `prev{N}_time_z` | 過去 N 走の走破タイム真 Z スコア | float | 馬場状態を除去した純粋なパフォーマンス偏差値 |

**真 Z スコアの計算**: 同日同コースのレースタイム平均・標準偏差で標準化し、馬場の影響を除去。

### 2.3 上がり 3 ハロン

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `prev{N}_last_3f` | 過去 N 走の上がり 3F タイム（秒） | float | 直線での末脚能力 |
| `prev{N}_last_3f_z` | 上がり 3F の Z スコア | float | 同レース内での相対評価 |
| `avg_last_3f_5` | 過去 5 走の上がり 3F 平均 | float | |
| `best_last_3f_5` | 過去 5 走の最速上がり 3F | float | |

### 2.4 着差

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `prev{N}_margin` | 過去 N 走の勝ち馬からの着差（秒） | float |

### 2.5 集約指標

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `ewm_speed_figure` | スピード指数の EWM（指数移動平均） | float | 直近レースほど重みが大きい。`halflife=3` 等 |
| `ewm_last_3f` | 上がり 3F の EWM | float | 同上 |
| `career_avg_speed_figure` | キャリア通じたスピード指数平均 | float | 能力のベースライン |
| `career_avg_last_3f` | キャリア通じた上がり 3F 平均 | float | |
| `form_score` | 近走調子スコア | float | EWM ベースの総合指標 |

---

## 3. 脚質・展開系

### 3.1 脚質判定

JRA-VAN 方式でコーナー通過順位から分類:

- **逃げ**: 最終コーナー以外のいずれかで 1 位通過
- **先行**: 逃げ以外で最終コーナー 4 位以内
- **差し**: 上記以外で最終コーナー通過順位が頭数の 2/3 以内
- **追込**: それ以外

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `prev{N}_running_style` | 過去 N 走の脚質 | categorical |
| `prev{N}_position_score` | 過去 N 走の位置取りスコア | float |
| `avg_position_score_5` | 過去 5 走の位置取りスコア平均 | float |
| `primary_running_style` | 主戦脚質 | categorical |

**位置取りスコア**: コーナー通過順位を 0.0（先頭）〜1.0（最後方）に正規化。`(通過順位 - 1) / (出走頭数 - 1)`

### 3.2 ペース分析

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `prev{N}_first_3f` | 過去 N 走の前半 3F タイム | float | |
| `prev{N}_pace_balance` | 前半 3F / 後半 3F の比率 | float | < 1.0 はスローペース |
| `prev{N}_lpi` | ラップ偏差指数 | float | < 0.15 = 持続型、> 0.30 = 瞬発型 |
| `prev{N}_pci` | ペースクラス指数 | float | |
| `pace_adaptability` | ペース順応指数 | float | 異なるペースへの適応力 |

### 3.3 展開予想

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `num_front_runners` | 逃げ馬の数 | int | レース出走メンバーから算出 |
| `expected_pace` | 予想ペース | categorical | ハイ/ミドル/スロー |
| `pace_bias_score` | 展開バイアススコア | float | 脚質構成が自分に有利かどうか |

---

## 4. 条件別適性

### 4.1 馬場適性（最重要カテゴリ）

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `turf_win_rate` | 芝での好走率 | float |
| `dirt_win_rate` | ダートでの好走率 | float |
| `going_good_perf` | 良馬場でのパフォーマンス偏差 | float |
| `going_yielding_perf` | 稍重でのパフォーマンス偏差 | float |
| `going_soft_perf` | 重馬場でのパフォーマンス偏差 | float |
| `going_heavy_perf` | 不良馬場でのパフォーマンス偏差 | float |

### 4.2 距離適性

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `sprint_perf` | スプリント（1000-1400m）適性 | float | 好走率・平均パフォーマンス |
| `mile_perf` | マイル（1600m）適性 | float | |
| `middle_perf` | 中距離（1800-2200m）適性 | float | |
| `long_perf` | 長距離（2400m+）適性 | float | |
| `this_distance_perf` | 今回距離での好走率 | float | |
| `distance_experience` | 今回距離での出走回数 | int | |

### 4.3 コース適性

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `this_course_win_rate` | 今回競馬場での好走率 | float |
| `left_turn_perf` | 左回りでの成績 | float |
| `right_turn_perf` | 右回りでの成績 | float |
| `western_turf_perf` | 洋芝での成績 | float |
| `condition_experience` | 同条件での出走回数 | int |
| `condition_place_rate` | 同条件での入着率 | float |

---

## 5. 血統適性

**重要**: one-hot エンコーディングは次元爆発するため不適。以下の集約手法を使う。

### 5.1 種牡馬系統

JRDB コード表 `keito_code.txt` でサイアーラインのグループ分類を利用。

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `sire_line_code` | 種牡馬系統コード | categorical | サンデーサイレンス系、ロベルト系等 |
| `broodmare_sire_line_code` | 母父系統コード | categorical | |

### 5.2 種牡馬の条件別産駒成績

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `sire_turf_win_rate` | 種牡馬産駒の芝勝率 | float |
| `sire_dirt_win_rate` | 種牡馬産駒のダート勝率 | float |
| `sire_distance_affinity` | 種牡馬産駒の今回距離カテゴリ好走率 | float |
| `sire_going_affinity` | 種牡馬産駒の今回馬場状態好走率 | float |
| `sire_course_affinity` | 種牡馬産駒の今回競馬場好走率 | float |
| `sire_avg_last_3f` | 種牡馬産駒の平均上がり 3F | float |

### 5.3 母父の条件別成績

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `broodmare_sire_surface_score` | 母父産駒の芝 / ダート適性スコア | float |
| `broodmare_sire_going_score` | 母父産駒の馬場状態適性スコア | float |
| `broodmare_sire_distance_score` | 母父産駒の距離適性スコア | float |

### 5.4 血統クラスタリング（Phase 5 向け）

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `bloodline_cluster_id` | KMeans 等でクラスタリングした血統グループ ID | int | 種牡馬+母父の組み合わせパターン |

---

## 6. 関係者成績

### 6.1 騎手

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `jockey_win_rate` | 全体勝率 | float |
| `jockey_top3_rate` | 全体 TOP3 率 | float |
| `jockey_course_win_rate` | 今回競馬場での勝率 | float |
| `jockey_distance_win_rate` | 今回距離カテゴリでの勝率 | float |
| `jockey_going_win_rate` | 今回馬場状態での勝率 | float |
| `jockey_position_bias` | 騎手の位置取りバイアス | float | 平均位置取りスコア。低い = 前に行く傾向 |
| `jockey_course_affinity` | 騎手×コース相性スコア | float | |
| `is_apprentice` | 減量騎手フラグ | bool | |

### 6.2 調教師

| 変数名 | 定義 | 型 |
|--------|------|-----|
| `trainer_win_rate` | 全体勝率 | float |
| `trainer_course_win_rate` | 今回競馬場での勝率 | float |
| `trainer_debut_win_rate` | 新馬戦勝率 | float |
| `trainer_layoff_win_rate` | 休み明け成績 | float |
| `trainer_stable` | 所属（栗東 / 美浦） | categorical | |

### 6.3 シナジー

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `jockey_trainer_win_rate` | 騎手×調教師の組み合わせ勝率 | float | サンプル数が少ない場合は全体平均にスムージング |

---

## 7. 調教データ

JRDB 独自指数 + 追い切りデータ。差別化要因として重要。

### 7.1 JRDB 指数

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `idm` | JRDB 総合指数 | float | KYI から取得 |
| `jrdb_jockey_index` | 騎手指数 | float | |
| `jrdb_info_index` | 情報指数 | float | |
| `jrdb_training_index` | 調教指数 | float | |
| `jrdb_stable_index` | 厩舎指数 | float | |
| `jrdb_fitness_index` | 仕上指数 | float | TYB（直前情報）で更新される場合あり |

### 7.2 追い切りデータ（Phase 5 向け）

CYB/CHA ファイルから取得。`pd.merge_asof` でレース直前の追い切りを紐付け。

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `workout_course` | 調教コース | categorical | 坂路/CW/ポリ等。コード表 `cyokyo_course_code.txt` |
| `workout_time` | 追い切りタイム | float | |
| `workout_last_1f` | ラスト 1F タイム | float | 偏差値 60 以上で好走傾向 |
| `workout_intensity` | 追い状態 | categorical | 一杯/強め/馬なり等。コード表 `oi_code.txt` |
| `workout_companion_result` | 併せ馬結果 | categorical | 先着/同入/遅れ |

**注意**: 外厩での調整は記録に表れない。調教データ単体の予測力は限定的で、他の特徴量と組み合わせて効果を発揮する。

---

## 8. レース内相対評価

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `race_relative_rank` | レース内での指数ランク | int | 出走メンバー内の順位 |
| `race_relative_z` | レース内偏差 | float | メンバー内での標準化スコア |
| `field_strength` | メンバーレベル | float | 出走馬全体の平均能力値 |
| `odds_gap` | オッズ断層 | float | 上位人気馬への資金集中度。レースの堅さ指標 |
| `market_share` | 市場シェア | float | 各馬の人気集中度 |

**注意**: `odds_gap` と `market_share` は前日オッズ（JRDB 基準オッズ）ベース。当日確定オッズを使うとリークになる。

---

## 9. 馬体情報

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `body_weight` | 馬体重（kg） | float | **当日発走 60 分前に発表**。前日予測では使えない |
| `body_weight_change` | 馬体重増減（前走比） | float | 同上 |
| `weight_ratio` | 馬体重比（斤量 / 馬体重） | float | 12.5% 超で成績低下傾向。同上 |
| `sex_code` | 性別コード | categorical | 牡/牝/セン |

**リーク注意**: 前日予測モデルでは `body_weight` 系を特徴量に含めてはならない。当日直前予測モデル専用。

---

## 10. レーティング系（Phase 5 向け）

| 変数名 | 定義 | 型 | 注意点 |
|--------|------|-----|--------|
| `glicko2_rating` | Glicko-2 レーティング | float | 初期値 1500。`glicko2-py` ライブラリ使用 |
| `glicko2_rd` | Glicko-2 RD（信頼度幅） | float | 初期値 350。小さいほど信頼度が高い |
| `glicko2_volatility` | Glicko-2 ボラティリティ | float | 初期値 0.06。強さの安定性 |
| `openskill_mu` | OpenSkill μ | float | |
| `integrated_rating` | 統合レーティング | float | Glicko-2 + OpenSkill を 0-100 に正規化 |

---

## 特徴量に関する横断的な注意事項

### リーク防止

以下の情報は **予測時点では利用できない** ため、特徴量に含めてはならない:

| 情報 | 利用可能時点 | 予測タイミング |
|------|-------------|---------------|
| 確定着順 | レース後 | ラベルとしてのみ使用 |
| 確定オッズ | 発走直前 | 前日予測では不可 |
| 馬体重 | 発走 60 分前 | 前日予測では不可 |
| 返し馬情報 | レース直前 | 不可 |
| パドック評価 | レース 1-2 時間前 | 前日予測では不可 |
| 当日の走破タイム | レース後 | 不可 |

### 欠損値の扱い

- 新馬・未出走の馬: 過去走系の特徴量が全て欠損。LightGBM は欠損値を自然に処理可能
- 初コース・初距離: 条件別適性が欠損。全体平均でフォールバック
- 騎手変更: 騎手×調教師シナジーが欠損。全体平均でスムージング

### カテゴリ変数のエンコーディング

| 変数 | 方式 | 理由 |
|------|------|------|
| 競馬場 | LightGBM native categorical | カテゴリ数が 10 で適度 |
| 馬場状態 | 序数エンコーディング（良=0, 稍重=1, 重=2, 不良=3） | 含水率に順序性がある |
| クラス | 序数エンコーディング（新馬=0 ... G1=9） | レベルに順序性がある |
| 脚質 | LightGBM native categorical | 順序性がない |
| 種牡馬系統 | LightGBM native categorical | カテゴリ数 20-30 |
| 性別 | LightGBM native categorical | |
