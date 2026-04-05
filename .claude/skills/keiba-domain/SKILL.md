---
name: keiba-domain
description: >-
  JRA中央競馬予想システムのドメイン制約とコンテキスト読み込み。
  競馬関連のコード作成・レビュー・特徴量設計・DB設計・テスト作成時に自動適用。
  keiba/ パッケージの開発全般で参照。
  競馬、馬、騎手、芝、ダート、血統、馬券、JRA、JRDBに言及するタスクで自動適用。
---

# 競馬（JRA）ドメインスキル

## Step 1: コンテキストの読み込み（最初に必ず実行）

競馬関連のコードを変更する前に、以下のドキュメントとコードを読み込んでコンテキストを取得すること。

### 必ず読むドキュメント

タスクの種類に応じて、以下のドキュメントを Read ツールで読み込む:

| タスクの種類 | 読むべきドキュメント |
|-------------|-------------------|
| 全般（最低限） | `docs/keiba/glossary.md`（命名規則） |
| 特徴量の追加・変更 | `docs/keiba/02-feature-catalog.md` |
| Enum・ドメインモデル変更 | `docs/keiba/01-jra-structure.md` |
| JRDB パーサー・DB 変更 | `docs/keiba/03-data-sources.md` |
| モデル設計変更 | `docs/keiba/04-model-design.md` |
| 戦略パラメータ変更 | `docs/keiba/05-strategy-and-market.md` |
| コードレビュー | `docs/keiba/06-pitfalls.md` |

### 必ず読むソースコード

変更対象に応じて、以下の関連コードを読み込む（keiba パッケージ構築後）:

| 変更対象 | 読むべきコード |
|---------|---------------|
| 特徴量 | `src/providence/keiba/features/pipeline.py` |
| Enum | `src/providence/keiba/domain/enums.py` |
| JRDB パーサー | `src/providence/keiba/scraper/jrdb.py` |
| DB スキーマ | `src/providence/keiba/database/tables.py` |
| Predictor | `src/providence/keiba/predictor.py` |
| 共通確率モデル | `src/providence/probability/plackett_luce.py`（or `core/` 配下） |
| 共通戦略 | `src/providence/strategy/types.py`（or `core/` 配下） |

## Step 2: ドメイン制約を適用してコードを書く

### リーク防止ルール

予測時点で使えない情報を特徴量に含めてはならない。

**前日予測モデルで使用禁止**:
- 確定着順（ラベルとしてのみ使用）
- 確定オッズ（発走直前に確定。JRDB 基準オッズは可）
- 馬体重・馬体重増減（発走 60 分前に発表）
- 返し馬・パドック評価
- 当日の走破タイム

**特徴量計算時の原則**:
- 必ず `as_of_date` パラメータでカットオフし、未来のデータを除外
- 騎手・調教師の通算成績も対象日時点までの集計にする
- `assert_no_leakage` テストを必ず実装

### モデル分離原則

- **芝とダートは必ず別セグメント**で学習。混合モデルは禁止
- **新馬戦は専用モデル**（コールドスタート問題。過去走データなし）
- **障害レースは初期は除外**

### ラベル設計

ランキング学習（LambdaRank）用: 1着=3, 2着=2, 3着=1, 4着以下=0

### 控除率定数

```python
RAKE_RATES = {
    "win": 0.20,       # 単勝
    "place": 0.20,     # 複勝
    "quinella": 0.225,  # 馬連
    "wide": 0.225,      # ワイド
    "bracket_quinella": 0.225,  # 枠連
    "exacta": 0.25,     # 馬単
    "trio": 0.25,       # 3連複
    "trifecta": 0.275,  # 3連単
    "win5": 0.30,       # WIN5
}
```

### 命名規則

- テーブル名: `keiba_` プレフィックス（例: `keiba_races`, `keiba_horses`）
- 変数名: `docs/keiba/glossary.md` に準拠
- 新しいドメイン用語を導入したら glossary.md に追記

### JRDB 連携キー

- **レースキー**: レースを一意に特定
- **血統登録番号**: 馬を一意に特定（`horse_registration_id`）
- **騎手コード / 調教師コード**: 関係者を一意に特定
- ファイル形式: 固定長テキスト、CP932 エンコーディング

## Step 3: ドキュメントを同時に更新

コード変更後、`.claude/skills/domain-doc-sync/SKILL.md` のマッピングに従い、対応するドキュメントを更新する。

## 詳細リファレンス

- JRA 競技構造: `docs/keiba/01-jra-structure.md`
- 特徴量カタログ: `docs/keiba/02-feature-catalog.md`
- データソースと JRDB 仕様: `docs/keiba/03-data-sources.md`
- モデル設計: `docs/keiba/04-model-design.md`
- 馬券戦略: `docs/keiba/05-strategy-and-market.md`
- 罠と対策: `docs/keiba/06-pitfalls.md`
- 用語集: `docs/keiba/glossary.md`
