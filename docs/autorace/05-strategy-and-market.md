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
| `min_probability` | 0.0 | 最小モデル確率 |
| `max_candidates` | 12 | 最大候補数 |
| `allowed_ticket_types` | None（全券種） | 対象券種の制限 |

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

## 8. 評価モード

`src/providence/strategy/types.py` の `EvaluationMode`:

| モード | 用途 |
|--------|------|
| `live` | 本番運用 |
| `fixed` | 固定パラメータでの再現テスト |
| `walk-forward` | ウォークフォワード検証 |
