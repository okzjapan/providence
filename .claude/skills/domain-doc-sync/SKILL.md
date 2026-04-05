---
name: domain-doc-sync
description: >-
  ドメイン関連コードの変更時にドキュメントの同時更新を強制する。
  オートレース（autorace/）または競馬（keiba/）のコードを作成・変更した際に自動適用。
  特徴量の追加、Enum変更、DBスキーマ変更、戦略パラメータ変更、モデル設計変更時に必ず参照。
---

# ドメインドキュメント同期スキル

## 原則

**コードを変更したら、対応するドメインドキュメントを必ず同時に更新する。**
ドキュメントの更新なしにコード変更をコミットしてはならない。

## 変更検知 → ドキュメント更新マッピング

以下の表に従い、コード変更に対応するドキュメントを特定して更新する。

### オートレース

| コード変更箇所 | 更新すべきドキュメント |
|---------------|---------------------|
| `src/providence/domain/enums.py` の TrackCode, TrackCondition, RiderRank, EntryStatus, RaceStatus, TicketType, Grade | `docs/autorace/01-autorace-structure.md`, `docs/autorace/glossary.md` |
| `src/providence/features/*.py` の特徴量追加・変更・削除 | `docs/autorace/02-feature-catalog.md`, `docs/autorace/glossary.md` |
| `src/providence/features/pipeline.py` のカテゴリマップ・除外列 | `docs/autorace/02-feature-catalog.md` |
| `src/providence/scraper/autorace_jp.py` または `oddspark.py` | `docs/autorace/03-data-sources.md` |
| `src/providence/scraper/schemas.py` のスキーマ変更 | `docs/autorace/03-data-sources.md` |
| `src/providence/database/tables.py` | `docs/autorace/03-data-sources.md` |
| `src/providence/model/trainer.py`, `predictor.py` | `docs/autorace/04-model-design.md` |
| `src/providence/probability/plackett_luce.py` | `docs/autorace/04-model-design.md` |
| `src/providence/strategy/*.py` のパラメータ・ロジック変更 | `docs/autorace/05-strategy-and-market.md` |
| `src/providence/strategy/confidence.py` | `docs/autorace/04-model-design.md` |

### 競馬（JRA）

| コード変更箇所 | 更新すべきドキュメント |
|---------------|---------------------|
| `src/providence/keiba/domain/enums.py` | `docs/keiba/01-jra-structure.md`, `docs/keiba/glossary.md` |
| `src/providence/keiba/features/*.py` の特徴量追加・変更・削除 | `docs/keiba/02-feature-catalog.md`, `docs/keiba/glossary.md` |
| `src/providence/keiba/scraper/*.py` | `docs/keiba/03-data-sources.md` |
| `src/providence/keiba/database/tables.py` | `docs/keiba/03-data-sources.md` |
| `src/providence/keiba/predictor.py` | `docs/keiba/04-model-design.md` |
| `src/providence/keiba/features/pipeline.py` | `docs/keiba/02-feature-catalog.md` |
| 競馬用戦略パラメータの変更 | `docs/keiba/05-strategy-and-market.md` |

### 共通基盤

| コード変更箇所 | 更新すべきドキュメント |
|---------------|---------------------|
| `src/providence/core/model/` | `docs/autorace/04-model-design.md` と `docs/keiba/04-model-design.md` の両方 |
| `src/providence/core/strategy/` | `docs/autorace/05-strategy-and-market.md` と `docs/keiba/05-strategy-and-market.md` の両方 |
| `src/providence/core/probability/` | 両方の `04-model-design.md` |
| `src/providence/core/backtest/` | 必要に応じて両方の `06-pitfalls.md` |

## 更新手順

### 1. 変更対象のドキュメントを特定

上記マッピング表を参照し、今回のコード変更が影響するドキュメントを列挙する。

### 2. ドキュメントを読み込む

対象ドキュメントを Read ツールで読み込み、現在の内容を確認する。

### 3. 差分を反映

以下のパターンに従って更新:

- **特徴量の追加**: `02-feature-catalog.md` の該当カテゴリに新しい行を追加。変数名・定義・型・計算式・注意点を記載
- **特徴量の削除**: `02-feature-catalog.md` から該当行を削除
- **特徴量の変更**: `02-feature-catalog.md` の計算式・定義を更新
- **Enum 値の追加/変更**: `01-*-structure.md` の該当テーブルと `glossary.md` を更新
- **DB カラムの追加**: `03-data-sources.md` のスキーマセクションを更新
- **新しいドメイン用語**: `glossary.md` に日本語/英語/コード変数名の対応を追記
- **戦略パラメータの変更**: `05-strategy-and-market.md` の該当テーブルを更新

### 4. 整合性チェック

更新後に以下を確認:
- `glossary.md` の変数名がコード上の実際の変数名と一致するか
- `02-feature-catalog.md` の計算式がコードの実装と一致するか
- 相互参照リンクが壊れていないか

## よくあるケース別ガイド

### 新しい特徴量を追加する場合

1. `features/*.py` に実装
2. `docs/{sport}/02-feature-catalog.md` の該当カテゴリに行を追加
3. `docs/{sport}/glossary.md` に変数名の対応を追記
4. テストを追加
5. コミット

### 新しい Enum 値を追加する場合

1. `domain/enums.py` に値を追加
2. `docs/{sport}/01-*-structure.md` の該当テーブルに行を追加
3. `docs/{sport}/glossary.md` に対応を追記
4. コミット

### 戦略パラメータのデフォルト値を変更する場合

1. `strategy/types.py` の `StrategyConfig` を変更
2. `docs/{sport}/05-strategy-and-market.md` のパラメータテーブルを更新
3. コミット
