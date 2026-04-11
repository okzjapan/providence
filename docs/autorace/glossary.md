# 用語集

オートレースドメインの日本語用語、英語表現、コード上の変数名の対応表。
新しいドメイン用語をコードに導入したら、このドキュメントに追記すること。

---

## レース場

| 日本語 | 英語 | コード | 場 ID |
|--------|------|--------|-------|
| 川口 | Kawaguchi | `TrackCode.KAWAGUCHI` | 2 |
| 伊勢崎 | Isesaki | `TrackCode.ISESAKI` | 3 |
| 浜松 | Hamamatsu | `TrackCode.HAMAMATSU` | 4 |
| 飯塚 | Iizuka | `TrackCode.IIZUKA` | 5 |
| 山陽 | Sanyo | `TrackCode.SANYO` | 6 |

## 走路状態

| 日本語 | 英語 | コード | エンコード値 |
|--------|------|--------|-------------|
| 良 | good | `TrackCondition.GOOD` | 0 |
| 湿 | wet | `TrackCondition.WET` | 1 |
| 重 | heavy | `TrackCondition.HEAVY` | 2 |
| 斑 | mixed | `TrackCondition.MIXED` | 3 |

## グレード

| 日本語 | 英語 | コード | エンコード値 |
|--------|------|--------|-------------|
| 普通 | normal | `Grade.NORMAL` | 0 |
| GII | grade 2 | `Grade.GII` | 1 |
| GI | grade 1 | `Grade.GI` | 2 |
| SG | super grade | `Grade.SG` | 3 |

## 車券

| 日本語 | 英語 | コード | 控除率 | autorace.jp キー |
|--------|------|--------|--------|-----------------|
| 単勝 | win | `TicketType.WIN` | 25% | `tns` |
| 複勝 | place | `TicketType.PLACE` | 25% | `fns` |
| 2 連単 | exacta | `TicketType.EXACTA` | 25% | `rtw` |
| 2 連複 | quinella | `TicketType.QUINELLA` | 25% | `rfu` |
| ワイド | wide | `TicketType.WIDE` | 25% | `wid` |
| 3 連単 | trifecta | `TicketType.TRIFECTA` | 25% | `srt` |
| 3 連複 | trio | `TicketType.TRIO` | 25% | `sfu` |

## 選手ランク

| 日本語 | 英語 | コード |
|--------|------|--------|
| S 級 | S rank | `RiderRank.S` |
| A 級 | A rank | `RiderRank.A` |
| B 級 | B rank | `RiderRank.B` |

## 出走状態

| 日本語 | 英語 | コード |
|--------|------|--------|
| 出走 | racing | `EntryStatus.RACING` |
| 取消 | cancelled | `EntryStatus.CANCELLED` |
| 落車 | fell | `EntryStatus.FELL` |
| 失格 | disqualified | `EntryStatus.DISQUALIFIED` |

## レース成立状態

| 日本語 | 英語 | コード |
|--------|------|--------|
| 正常 | normal | `RaceStatus.NORMAL` |
| 一部取消 | partial cancel | `RaceStatus.PARTIAL_CANCEL` |
| 不成立 | void | `RaceStatus.VOID` |

## レース・出走

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 枠番 | post position | `post_position` | 1-8 |
| ハンデ | handicap | `handicap_meters` | メートル単位。0-110 |
| 試走タイム | trial time | `trial_time` | 秒単位 |
| 平均試走タイム | average trial time | `avg_trial_time` | |
| 試走偏差 | trial deviation | `trial_deviation` | |
| レーススコア | race score | `race_score` | |
| 着順 | finish position | `finish_position` | |
| レースタイム | race time | `race_time` | 秒単位 |
| スタートタイミング | start timing | `start_timing` | |
| 事故コード | accident code | `accident_code` | |
| 距離 | distance | `distance` | 原則 3100m |

## 選手

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 選手 / ライダー | rider | `rider_id` | |
| 登録番号 | registration number | `registration_number` | ユニーク識別子 |
| 所属場 | home track | `home_track_id` | |
| 世代（期） | generation | `generation` | |

## 特徴量

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 試走順位 | trial time rank | `trial_time_rank` | レース内 |
| 最速試走との差 | diff from best trial | `trial_time_diff_from_best` | |
| レース内平均試走 | field average trial | `field_avg_trial` | |
| レース内試走標準偏差 | field trial std | `field_trial_std` | |
| 30 日平均試走 | 30-day avg trial | `avg_trial_time_30d` | |
| 90 日平均試走 | 90-day avg trial | `avg_trial_time_90d` | |
| 試走トレンド | trial time trend | `trial_time_trend` | 正=改善 |
| 試走 vs 90 日平均 | trial vs 90d avg | `trial_vs_avg_90d` | |
| 直近 10 走勝率 | win rate (last 10) | `win_rate_10` | |
| 直近 10 走 2 着内率 | top2 rate (last 10) | `top2_rate_10` | |
| 直近 10 走 3 着内率 | top3 rate (last 10) | `top3_rate_10` | |
| 直近 30 走勝率 | win rate (last 30) | `win_rate_30` | |
| 直近 30 走 3 着内率 | top3 rate (last 30) | `top3_rate_30` | |
| 直近 10 走平均着順 | avg finish (last 10) | `avg_finish_10` | |
| 直近 30 走平均着順 | avg finish (last 30) | `avg_finish_30` | |
| 着順標準偏差 | finish std (last 10) | `finish_std_10` | |
| モメンタム | momentum | `momentum` | 直近 5 走の着順トレンド。正=改善 |
| 前走からの日数 | days since last race | `days_since_last_race` | |
| 通算レース数 | total races | `total_races` | |
| ハンデ順位 | handicap rank | `handicap_rank` | レース内 |
| 出走頭数 | field size | `field_size` | |
| ハンデ vs 試走 | handicap vs trial | `handicap_vs_trial` | 正=ハンデの割に試走良好 |
| レース内平均勝率 | field avg win rate | `field_avg_win_rate` | |
| レース内最高勝率 | field max win rate | `field_max_win_rate` | |
| レース内平均着順平均 | field avg finish | `field_avg_finish` | |
| フォーム順位 | form rank in field | `form_rank_in_field` | |
| 相対勝力 | relative win strength | `relative_win_strength` | |
| トップライバルとの差 | gap to top rival | `gap_to_top_rival` | |
| 自前計算試走偏差 | computed trial deviation | `computed_trial_deviation` | 良走路・直近10走・最低5走 |
| 試走偏差トレンド | trial deviation trend | `trial_deviation_trend` | 正=偏差縮小 |
| 予想競走タイム | predicted race time | `predicted_race_time` | `trial_time + computed_trial_deviation` |
| 予想タイム順位 | predicted race time rank | `predicted_race_time_rank` | レース内 |
| 予想タイム vs ハンデ | predicted vs handicap | `predicted_vs_handicap` | 正=ハンデの割にタイム良好 |
| 試走 Zスコア | trial time z-score | `trial_time_zscore` | レース内偏差値。std=0 時は 0 |
| 平均ST（10走） | avg start timing (last 10) | `avg_start_timing_10` | |
| 平均ST（30走） | avg start timing (last 30) | `avg_start_timing_30` | |
| ST安定性 | start timing consistency | `start_timing_consistency` | 低い=安定 |
| ホーム場フラグ | is home track | `is_home_track` | 0/1 |
| 選手年齢 | rider age | `rider_age` | `race_date.year - birth_year` |
| フィールドハンデ差 | handicap range in field | `handicap_range_in_field` | max - min |
| ハンデグループ | handicap group | `handicap_group` | 0=0m, 1=10m, 2=20-30m, 3=40-50m, 4=60m+ |
| 湿走路フラグ | is wet condition | `is_wet_condition` | 0/1 |
| 湿重走路3着内率 | rider wet top3 rate | `rider_wet_top3_rate` | 5 走以上 |
| 場別勝率 | rider track win rate | `rider_track_win_rate` | 5 走以上 |
| 湿重走路勝率 | rider wet win rate | `rider_wet_win_rate` | 5 走以上 |

## 戦略

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 期待値 | expected value | `expected_value` | `p × odds - 1` |
| 信頼度 | confidence score | `confidence_score` | 0-1 |
| 判断時刻 | judgment time | `judgment_time` | |
| Kelly 分数 | fractional Kelly | `fractional_kelly` | コード既定 0.25。V013 採用設定では **0.05**（0.10以上は赤字） |
| 推奨額 | recommended bet | `recommended_bet` | 円。100 円単位 |
| 最大合計賭け金 | max total stake | `max_total_stake` | 既定 10,000 円 |
| 取り込みバッチ ID | ingestion batch ID | `ingestion_batch_id` | オッズの時系列管理用 |

## オッズ・払い戻し

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 市場オッズ | market odds | `odds_value` | `odds_snapshot` テーブル |
| 払い戻し | payout / refund | `payout_value` | `ticket_payouts` テーブル |
| 基準オッズ | base odds | — | KYI から取得（JRDB、競馬用） |

## V013 以降の追加用語

| 日本語 | 英語 | 変数名 | 備考 |
|--------|------|--------|------|
| 前走人気順位 | previous odds rank | `odds_rank_prev1`〜`prev5` | 前走での単勝オッズに基づく人気順位 |
| 市場エラースコア | value score | `value_score` | `max(0, win_odds_rank - finish_position)` |
| Model B | value model | `value_model_b` | 市場エラーを予測する回帰モデル |
| 能力 vs 市場 | ability vs market | `ability_vs_market` | Model A 順位 - 市場人気順位 |
| 統合スコア | combined score | — | `α × Model_A + (1-α) × Model_B` |
| FORM_SCORE | form score | `form_score` | 直近5走の重み付き着順平均（0.5指数減衰） |
| EWM | exponential weighted mean | `ewm_race_time` 等 | 指数加重平均 |
| 人気上回り率 | beat odds rate | `beat_odds_rate` | 直近5走で着順が人気を上回った割合 |
| 実効予測タイム | effective predicted time | `effective_predicted_time` | `predicted_race_time + handicap_seconds` |
| 控除率 | takeout rate | `PAYOUT_RATE` | オートレース一律25%。`PAYOUT_RATE=0.75` |
