# 用語集

競馬ドメインの日本語用語、英語表現、コード上の変数名の対応表。
新しいドメイン用語をコードに導入したら、このドキュメントに追記すること。

---

## レース・コース

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 競馬場 | racecourse | `racecourse_id` | 10 場 |
| 芝 | turf | `surface = "turf"` | |
| ダート | dirt | `surface = "dirt"` | |
| 距離 | distance | `distance` | メートル単位 |
| 馬場状態 | going | `going` | 良/稍重/重/不良 |
| 良 | good / firm | `going = "good"` | |
| 稍重 | good to yielding | `going = "yielding"` | |
| 重 | soft | `going = "soft"` | |
| 不良 | heavy | `going = "heavy"` | |
| 内回り | inner course | `course_type = "inner"` | |
| 外回り | outer course | `course_type = "outer"` | |
| 直線 | straight | `course_type = "straight"` | 新潟千直等 |
| 洋芝 | western turf | `turf_type = "western"` | 札幌・函館 |
| 野芝 | japanese turf | `turf_type = "japanese"` | その他 8 場 |
| 開催日数 | race day number | `race_day_number` | トラックバイアスの代理変数 |
| 高低差 | elevation difference | `elevation_diff` | メートル単位 |
| 直線距離 | straight length | `straight_length` | メートル単位 |
| 左回り | left-handed | `direction = "left"` | |
| 右回り | right-handed | `direction = "right"` | |

## クラス・グレード

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 新馬 | maiden (debut) | `race_class = "maiden"` | |
| 未勝利 | maiden (non-winner) | `race_class = "non_winner"` | |
| 1 勝クラス | 1-win class | `race_class = "class_1"` | 旧 500 万下 |
| 2 勝クラス | 2-win class | `race_class = "class_2"` | 旧 1000 万下 |
| 3 勝クラス | 3-win class | `race_class = "class_3"` | 旧 1600 万下 |
| オープン | open | `race_class = "open"` | |
| リステッド | listed | `race_class = "listed"` | |
| G3 | grade 3 | `race_class = "g3"` | |
| G2 | grade 2 | `race_class = "g2"` | |
| G1 | grade 1 | `race_class = "g1"` | |

## 券種

| 日本語 | 英語 | 変数名 | 控除率 |
|--------|------|--------|--------|
| 単勝 | win | `TicketType.WIN` | 20% |
| 複勝 | place | `TicketType.PLACE` | 20% |
| 枠連 | bracket quinella | `TicketType.BRACKET_QUINELLA` | 22.5% |
| 馬連 | quinella | `TicketType.QUINELLA` | 22.5% |
| ワイド | quinella place / wide | `TicketType.WIDE` | 22.5% |
| 馬単 | exacta | `TicketType.EXACTA` | 25% |
| 3 連複 | trio / trifecta place | `TicketType.TRIO` | 25% |
| 3 連単 | trifecta | `TicketType.TRIFECTA` | 27.5% |
| WIN5 | win5 | `TicketType.WIN5` | 30% |

## 馬・出走

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 馬番 | post position | `post_position` | 1-18 |
| 枠番 | draw / gate | `draw` | 1-8 |
| 斤量 | impost weight | `impost_weight` | kg |
| 馬体重 | body weight | `body_weight` | kg。当日発表 |
| 馬体重増減 | weight change | `body_weight_change` | 前走比。kg |
| 馬齢 | age | `age` | |
| 牡馬 | colt / male | `sex = "male"` | |
| 牝馬 | filly / female | `sex = "female"` | |
| セン馬 | gelding | `sex = "gelding"` | |
| 出走取消 | scratched | `entry_status = "scratched"` | |
| 競走中止 | pulled up | `entry_status = "pulled_up"` | |
| 失格 | disqualified | `entry_status = "disqualified"` | |
| 降着 | placed behind | `entry_status = "placed_behind"` | |

## 斤量制度

| 日本語 | 英語 | 変数名 |
|--------|------|--------|
| 馬齢重量 | weight for age | `weight_rule = "wfa"` |
| 定量 | set weight | `weight_rule = "set"` |
| 別定 | penalties | `weight_rule = "penalties"` |
| ハンデキャップ | handicap | `weight_rule = "handicap"` |

## パフォーマンス指標

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 走破タイム | race time | `race_time_sec` | 秒単位 |
| 上がり 3 ハロン | last 3 furlongs | `last_3f` | 秒単位。末脚能力 |
| 着差 | margin | `margin` | 秒単位 |
| スピード指数 | speed figure | `speed_figure` | 西田式 |
| 着順 | finish position | `finish_position` | |
| コーナー通過順位 | corner position | `corner_{N}_pos` | N = 1-4 |

## 脚質

| 日本語 | 英語 | 変数名 |
|--------|------|--------|
| 逃げ | front runner | `running_style = "front"` |
| 先行 | stalker / presser | `running_style = "stalker"` |
| 差し | closer | `running_style = "closer"` |
| 追込 | deep closer | `running_style = "deep_closer"` |
| 位置取りスコア | position score | `position_score` | 0.0（先頭）〜1.0（最後方） |

## 血統

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 種牡馬 | sire | `sire_id` | 父馬 |
| 母馬 | dam | `dam_id` | |
| 母父 | broodmare sire | `broodmare_sire_id` | 母方の父 |
| 系統 | sire line | `sire_line_code` | JRDB `keito_code.txt` |
| 血統登録番号 | breeding registration number | `horse_registration_id` | JRDB 連携キー |

## 調教

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 追い切り | workout / breeze | `workout_*` | |
| 坂路 | hill course | `workout_course = "hill"` | 栗東・美浦 |
| ウッドチップ（CW） | woodchip course | `workout_course = "cw"` | |
| ポリトラック | polytrack | `workout_course = "poly"` | |
| 一杯追い | full drive | `workout_intensity = "full"` | |
| 馬なり | hands and heels | `workout_intensity = "easy"` | |
| 併せ馬 | companion work | — | 先着/同入/遅れ |
| 栗東 | Ritto | `stable_location = "ritto"` | 西のトレセン |
| 美浦 | Miho | `stable_location = "miho"` | 東のトレセン |

## JRDB 指数

| 日本語 | 英語 | 変数名 |
|--------|------|--------|
| IDM | — | `idm` | JRDB 総合指数 |
| 騎手指数 | jockey index | `jrdb_jockey_index` |
| 情報指数 | info index | `jrdb_info_index` |
| 調教指数 | training index | `jrdb_training_index` |
| 厩舎指数 | stable index | `jrdb_stable_index` |
| 仕上指数 | fitness index | `jrdb_fitness_index` |

## JRDB 連携キー

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| レースキー | race key | `race_key` | レースを一意に特定 |
| 競走成績キー | result key | `result_key` | レース結果を一意に特定 |
| 騎手コード | jockey code | `jockey_code` | |
| 調教師コード | trainer code | `trainer_code` | |
