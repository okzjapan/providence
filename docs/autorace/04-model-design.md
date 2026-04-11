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

**V011 既定の重み**: ランク=0.40, 2着以内=0.30, 1着=0.15, Huber=0.15

**V012 既定の重み（6モデル）**: ランク=0.30, XE-NDCG=0.15, 2着以内=0.20, Focal=0.15, 1着=0.10, Huber=0.10

### 1.2.1 V011 パイプライン整合性改善

V011 で以下 3 点の内部一貫性を修正:

1. **Isotonic 入力の統一**: キャリブレーターの学習をアンサンブル統合済みスコアに変更（V010 では LambdaRank 単体スコアを使用）
2. **Kelly シナリオ確率の整合**: `enumerate_top3_scenarios` が `scenario_strengths`（キャリブレーション済み確率）を優先使用するよう変更。候補フィルタとKelly最適化で同一確率を参照する
3. **2連系補助ブレンド**: `binary_top2` モデル出力を `pair_blend_alpha`（検証セットで自動選択）で `exacta`/`quinella` 確率に混合。1-2着同時出現の相関を捕捉

`_blend_pair_ticket_probs()` は `main_probs` と `aux_strengths`（binary_top2 正規化出力）を `alpha` で線形混合する。`alpha=0` ならブレンドなし。

### 1.2.2 V012 6モデルアンサンブル + サンプル重み

V012 で追加された改善:

1. **XE-NDCG モデル**: `rank_xendcg` 目的関数。クロスエントロピーベースの滑らかなNDCG代替損失で確率キャリブレーションを改善
2. **Focal Value モデル**: Focal Loss (γ=2, α=0.75) カスタム目的関数。easy examples の損失を down-weight し、モデルに「難しいケース」（穴選手の好走）への注意を集中させる
3. **binary_win 安定化**: 学習率 0.05→0.01、正則化 (L1=L2=1.0)、num_leaves 63→31 で 4 イテレーション停止問題を解消
4. **サンプル重み付け**: `weight = 1/field_size` で全レースの学習貢献度を均一化。NDCG@1 が +0.013 改善
5. **Huber 収束改善**: num_boost_round 1000→2000 で収束近くまで学習
6. **複勝 (PLACE) 確率**: PL 確率計算に top-3 入り確率を追加。Kelly 最適化の `_ticket_hits` にも対応

### 1.2.3 V013 特徴量倍増（59 → 111）

V013 で追加された特徴量カテゴリ（`prev_race.py`, `context.py` に実装）:

1. **前走個別データ**: 前1走〜前5走の着順・競走タイム・ST・ハンデ・場・走路状態（+30）
2. **FORM_SCORE**: 直近5走の重み付き着順平均（0.5の指数減衰）。競馬AIで最も重要な特徴量に相当
3. **EWM スコア**: 競走タイム・STの指数加重平均。ローリング平均より直近に敏感
4. **前走タイム Z スコア**: 過去タイムの平均・標準偏差に対する各前走の標準化タイム
5. **レース文脈**: ST場内相対位置、休養日数セグメント、月、気温×走路状態交互作用

NDCG@1 が +1.7% 改善し、バックテスト利益が V011 比で大幅向上。V014（141特徴量）は過剰で効果なく、111が最適点。

### 1.3 Model B（Value モデル）

`src/providence/model/value_model.py` に実装。Model A とは独立に学習・推論する 2 段階モデル。

**目的**: 「市場がどこで間違えているか」を直接予測する。Model A が「誰が強いか」を予測するのに対し、Model B は「誰が市場に過小評価されているか」を予測する。

**ターゲット**: `value_score = max(0, win_odds_rank - finish_position)`

**特徴量**: V013 の 111 特徴量 + `win_odds_rank`, `ability_vs_market`（Model A 順位 - 市場人気順位）, `implied_prob`, `odds_entropy`

**統合**: `combined = α × Model_A_score + (1-α) × Model_B_score` で加重合成し、閾値以上のレースのみ投票。α=0.5, top70% フィルタで ROI が +28.6% → +54.5% に改善。

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

### 3.1.1 Isotonic Regression キャリブレーション

`src/providence/probability/calibration.py` の `IsotonicCalibrator`。

PL が出力した生の勝率を、検証セットで学習したノンパラメトリックな単調写像で補正する。温度スケーリング（1パラメータの対称補正）では捉えられない非対称な歪みを修正可能。

**適用フロー**:
1. スコア → PL(temperature) → 生の勝率 `p_raw[i]`
2. IsotonicRegression → キャリブレーション済み勝率 `p_cal[i]`
3. `p_cal[i]` を PL 互換の強度として `compute_all_ticket_probs_from_strengths()` で全券種確率を再計算

キャリブレーターは `isotonic_calibrator.pkl` としてモデルバージョンディレクトリに保存。Predictor がロード時に自動検出し適用する。

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
