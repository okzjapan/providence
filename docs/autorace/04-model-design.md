# モデル設計指針

## 1. アルゴリズム

### 1.1 LightGBM LambdaRank

現在のシステムは LightGBM の LambdaRank（ランキング学習）を採用。

- **目的関数**: `lambdarank`（NDCG 最適化）
- **ラベル**: 着順をスコアに変換（1 着=3, 2 着=2, 3 着=1, 4 着以下=0）
- **group**: 各レースの出走車数を指定

LambdaRank は同一レース内での相対評価を最適化するため、メンバーレベルが異なるレース間でも適切に学習できる。

### 1.2 4モデルアンサンブル

`--ensemble` フラグで4モデルを同時学習し、重み付き幾何平均で統合する。

| モデル | 目的関数 | ラベル | 役割 |
|--------|---------|--------|------|
| LambdaRank | `lambdarank` | `field_size - position` | 相対的ランキング品質 |
| Binary top-2 | `binary` | `position <= 2 ? 1 : 0` | 2連系・ワイドの精度 |
| Binary win | `binary` | `position == 1 ? 1 : 0` | 単勝精度 |
| Huber 回帰 | `huber` | `{1:8, 2:4, 3:2, 4:1, else:0}` | 外れ値にロバストな着順予測 |

**統合フロー** (`src/providence/model/ensemble.py`):
1. 各モデルの出力をレース内 softmax → 勝率分布に変換
2. 重み付き幾何平均: `p_combined[i] ∝ Π p_k[i]^w_k`
3. 正規化 → `log(p_combined)` を PL 層に `temperature=1.0` で渡す

**既定の重み**: ランク=0.40, 2着以内=0.30, 1着=0.15, Huber=0.15

### 1.3 Optuna によるハイパーパラメータ最適化

`src/providence/model/trainer.py` で Optuna を使ったチューニングが可能。

### 1.4 セグメント分割

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
- 単一モデル: `model.txt` + `metadata.json`
- アンサンブル: `lambdarank.txt`, `binary_top2.txt`, `binary_win.txt`, `huber.txt` + `ensemble_weights.json` + `metadata.json`
- `metadata.json` に `model_type` フィールド（`"lambdarank"` or `"ensemble"`）で判別
- `latest` ファイルで最新バージョンを管理
- `load_for_backtest` でバックテスト用の過去バージョンをロード可能
- `Predictor` が `model_type` を検出し、単一/アンサンブルを透過的に切り替え

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
