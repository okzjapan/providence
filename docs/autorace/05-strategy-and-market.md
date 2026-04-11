# 車券戦略と市場

## 1. 控除率

オートレースの控除率は全券種一律 **25%**。還元率は 75%。

ランダム購入した場合の期待回収率は 75%。利益を出すには 33% のエッジ（75% → 100%）が必要。

## 2. 期待値計算

`src/providence/strategy/expected_value.py`

```
expected_value = probability × odds - 1
```

- `probability`: Plackett-Luce から算出したモデル推定確率
- `odds`: 市場オッズ
- `expected_value > 0` は「控除率を考慮しても期待値がプラス」を意味

## 3. Kelly 最適化

`src/providence/strategy/kelly.py`

### 3.1 方式

Plackett-Luce の上位 3 着順の全順列をシナリオとし、各候補券の損益行列で相対重みを反復最適化する。

V011 以降、`RacePredictionBundle.scenario_strengths` が設定されている場合はそちらを優先使用する。`scenario_strengths` はキャリブレーション済み確率であり、候補フィルタと Kelly 最適化で同一の確率を参照することで一貫性を保つ。

### 3.2 パラメータ

| パラメータ | 既定値 | 説明 |
|-----------|--------|------|
| `fractional_kelly` | 0.25 | Kelly 分数。1/4 Kelly で分散を低減 |
| `min_bet_amount` | 100 | 最小賭け金（円）。100 円単位 |
| `max_total_stake` | 10,000 | 1 レースの最大合計賭け金 |
| `min_weight_threshold` | 0.01 | この重み未満の候補は切り捨て |

### 3.3 金額化プロセス

`src/providence/strategy/bankroll.py`

1. Kelly 最適化で各候補の相対重みを算出
2. 最小重みが 100 円になるようスケール
3. 100 円単位に切り捨て
4. 合計が `max_total_stake` を超えたら比例縮小して再丸め
5. 丸め後に 0 円になった候補は除外

**Phase 6 で変更**: bankroll（資金管理）は廃止。レース間で賭け金は独立。

## 4. 候補生成

`src/providence/strategy/candidates.py`

1. Plackett-Luce の予測確率と市場オッズを突合
2. 期待値（EV）と確率の閾値でフィルタ
3. EV 降順でソート
4. `max_candidates`（既定 12）件に制限

### フィルタ条件

| パラメータ | 既定値 | 説明 |
|-----------|--------|------|
| `min_expected_value` | 0.0 | 最小期待値 |
| `min_probability` | 0.0 | 最小モデル確率（全券種共通） |
| `min_probability_by_ticket` | None | 券種別最小モデル確率（設定時は `min_probability` より優先） |
| `min_odds` | None | 最小オッズ（これ未満の候補は除外） |
| `max_odds` | None | 最大オッズ（これ超の候補は除外） |
| `max_candidates` | 12 | 最大候補数 |
| `allowed_ticket_types` | None（全券種） | 対象券種の制限 |

**券種別 min_probability の推奨値**（`DEFAULT_MIN_PROBABILITY_BY_TICKET`）:

| 券種 | 閾値 | ランダムベースライン（8車） |
|------|------|--------------------------|
| 単勝 | 0.05 | 0.125 |
| 2連単 | 0.02 | 0.018 |
| 2連複 | 0.04 | 0.036 |
| ワイド | 0.05 | 0.107 |
| 3連単 | 0.006 | 0.003 |
| 3連複 | 0.02 | 0.018 |

## 5. スキップ理由

戦略実行で推奨が出ない場合の理由コード:

| `skip_reason` | 意味 |
|--------------|------|
| `no_market_odds` | 市場オッズが取得できていない |
| `low_confidence` | 信頼度が `min_confidence` 未満 |
| `no_positive_ev_candidates` | EV > 0 の候補がない |
| `rounded_below_minimum` | 金額正規化後に全候補が 0 円になった |

## 6. 運用フロー

`predict-next-race.md` に基づく 3 フェーズ運用:

### Phase 1: 発走 10 分前

- autorace.jp から出走表（Program）を取得
- 試走タイムを `race_entries` に反映
- Predictor で勝率を算出

### Phase 2: 発走 3 分前

- oddspark から最新オッズを取得
- `judgment_time` を設定し、Kelly 最適化を実行
- 推奨額を出力

### Phase 3: レース後

- 結果を取得し精算

## 7. ランク分類

予測結果を信頼度と期待値でランク付け:

| ランク | 条件 | 推奨アクション |
|--------|------|---------------|
| S | 高信頼度 + 高 EV | 強く推奨 |
| A-B | 中信頼度 + 正 EV | 推奨 |
| C-D | 低信頼度 or 低 EV | 注意して検討 |
| E | スキップ | 見送り |

## 8. 現在の採用戦略（V013）

### メイン: 単勝

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| `min_confidence` | 0.90 | conf>=0.90〜0.97 で利益正。0.90 がベット数最大（月10R） |
| `min_expected_value` | 0.40 | EV>=0.40 はモデル確率×オッズ>=1.4。高期待値レースに集中 |
| `max_candidates` | 2 | 1レース最大2本。通常は1本 |
| `fractional_kelly` | 0.05 | 0.10 以上で利益消滅。厳守 |
| `allowed_ticket_types` | `{WIN}` | 単勝のみ |

検証実績（2026-01〜04, 1,790R）: ROI +48.5%, profit +3,010円, 61R, 的中率33%

### サテライト（任意）: ワイド

V010 設定を V013 に適用: conf>=0.99, EV>=0.2, cand<=2, kelly=0.25。月1-2R、高配当上振れ狙い。

### Model B（Value フィルタ）

Model A の投票候補に対し、Model B の combined score で低品質ベットを排除する。

| パラメータ | 値 |
|-----------|-----|
| α | 0.5（Model A と B の均等配合） |
| フィルタ | combined score 上位 70% のレースのみ |

効果: ROI +28.6% → +54.5%（ベット数 99R → 60R）

### 重要な知見: kelly=0.05 の壁

kelly を 0.05 から 0.10 に上げると、Kelly 最適化がベット対象を拡大し（60R→224R）、低品質ベットが混入して利益が消滅する。Model B フィルタ適用後でも kelly=0.10 は赤字。kelly=0.05 は現行アーキテクチャの構造的限界。

---

## 9. 評価モード

`src/providence/strategy/types.py` の `EvaluationMode`:

| モード | 用途 |
|--------|------|
| `live` | 本番運用 |
| `fixed` | 固定パラメータでの再現テスト |
| `walk-forward` | ウォークフォワード検証 |
