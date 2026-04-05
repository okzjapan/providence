# モデル設計指針

## 1. アルゴリズム

### 1.1 LightGBM LambdaRank

現在のシステムは LightGBM の LambdaRank（ランキング学習）を採用。

- **目的関数**: `lambdarank`（NDCG 最適化）
- **ラベル**: 着順をスコアに変換（1 着=3, 2 着=2, 3 着=1, 4 着以下=0）
- **group**: 各レースの出走車数を指定

LambdaRank は同一レース内での相対評価を最適化するため、メンバーレベルが異なるレース間でも適切に学習できる。

### 1.2 Optuna によるハイパーパラメータ最適化

`src/providence/model/trainer.py` で Optuna を使ったチューニングが可能。

### 1.3 セグメント分割

現在のオートレースはセグメント分割なし（全レースを単一モデルで学習）。出走数が最大 8 車と少なく、芝/ダートのような根本的な分離が不要なため。

---

## 2. 特徴量パイプライン

### 2.1 学習時

```
DataLoader.load_race_dataset(end_date=cutoff)
  → FeaturePipeline.build_features(raw_df)
  → LightGBM 学習
  → ModelStore.save(model, metadata)
```

- `end_date` パラメータで同日リークを防止

### 2.2 予測時

```
Predictor.load_history(as_of_date)   # カットオフ日までの履歴のみ読込
  → build_features_for_race(entries, history)  # 対象レース + 過去のみで特徴量
  → LightGBM 推論 → スコア
  → compute_all_ticket_probs(scores, temperature)  # Plackett-Luce
```

- `load_history` は日付単位でキャッシュし、同日の複数レースで再利用

---

## 3. Plackett-Luce 確率モデル

`src/providence/probability/plackett_luce.py`

### 3.1 強度計算

```python
strengths = exp((score - max_score) / temperature)
```

`temperature` はモデルのメタデータに保存され、Predictor がロード時に読み込む。

### 3.2 各券種の確率計算

| 券種 | 計算方法 |
|------|---------|
| 単勝 | 強度の正規化 |
| 2 連単 | 逐次抽出（1 着目 → 残りから 2 着目） |
| 2 連複 | 2 連単の順序両方の和 |
| 3 連単 | 3 段の逐次抽出 |
| 3 連複 | 3 連単の 3 台の全順列の和 |
| ワイド | 3 連単から「上位 3 着に両方含まれる」確率の周辺化 |

**注意**: 複勝（`place`）は現在の `_ticket_hits` に未対応（Kelly 最適化のシナリオマトリクスに含まれていない）。

### 3.3 計算量

8 車の場合:
- 3 連単: 8×7×6 = 336 通り
- 全券種の確率計算は十分高速

---

## 4. バリデーション

### 4.1 時系列分割

- 訓練データの `end_date` を設定し、それ以前のデータのみで学習
- 検証・テストは `end_date` 以降のデータで実施
- ランダム分割は不可（時系列リーク）

### 4.2 評価指標

| 指標 | 用途 |
|------|------|
| NDCG | ランキング品質 |
| 回収率シミュレーション | 実戦的評価 |
| SHAP | 特徴量の重要度と解釈性 |

---

## 5. モデル管理

`src/providence/model/store.py`

- モデルは `data/models/<version>/` に保存
- 各バージョンに `model.txt`（LightGBM モデル）と `metadata.json`（特徴量リスト、temperature 等）を格納
- `latest` シンボリックリンクで最新バージョンを管理
- `load_for_backtest` でバックテスト用の過去バージョンをロード可能

---

## 6. 信頼度スコア

`src/providence/strategy/confidence.py`

```
confidence = 0.6 × spread_component + 0.4 × coverage
```

| 成分 | 計算 | 意味 |
|------|------|------|
| `spread_component` | `1 - exp(-max(score_spread, 0))` | スコアのばらつきが大きいほど予測に確信あり |
| `coverage` | 5 走以上の履歴がある選手の割合 | データが揃っているレースほど信頼度が高い |

信頼度が `min_confidence`（既定 0.1）未満のレースは `skip_reason = "low_confidence"` でスキップ。
