# domain-doc-sync リファレンス

## ドキュメントファイル一覧

### オートレース (`docs/autorace/`)

| ファイル | 内容 | 主な更新トリガー |
|---------|------|----------------|
| `README.md` | ドキュメントマップ | ファイル構成が変わった場合のみ |
| `01-autorace-structure.md` | 競技構造（場・車券・走路・選手・ハンデ等） | Enum変更、競技ルール変更 |
| `02-feature-catalog.md` | 特徴量カタログ | **最も頻繁に更新される** |
| `03-data-sources.md` | データソース・DBスキーマ | スクレイパー変更、テーブル変更 |
| `04-model-design.md` | モデル設計・確率モデル・信頼度 | アルゴリズム変更 |
| `05-strategy-and-market.md` | 戦略パラメータ・Kelly・運用フロー | 戦略ロジック変更 |
| `06-pitfalls.md` | 罠と対策チェックリスト | 新しい罠を発見した場合 |
| `glossary.md` | 用語集 | 新しいドメイン用語導入時 |

### 競馬 (`docs/keiba/`)

| ファイル | 内容 | 主な更新トリガー |
|---------|------|----------------|
| `README.md` | ドキュメントマップ | ファイル構成が変わった場合のみ |
| `01-jra-structure.md` | JRA競技構造 | Enum変更、競技ルール変更 |
| `02-feature-catalog.md` | 特徴量カタログ | **最も頻繁に更新される** |
| `03-data-sources.md` | JRDB仕様・DBスキーマ | パーサー変更、テーブル変更 |
| `04-model-design.md` | モデル設計・セグメント・Plackett-Luce | アルゴリズム変更 |
| `05-strategy-and-market.md` | 馬券戦略・市場効率性・Kelly | 戦略ロジック変更 |
| `06-pitfalls.md` | 罠と対策チェックリスト | 新しい罠を発見した場合 |
| `glossary.md` | 用語集 | 新しいドメイン用語導入時 |

## コードパス → ドメイン判定

| パスパターン | ドメイン |
|-------------|---------|
| `src/providence/autorace/` | オートレース |
| `src/providence/keiba/` | 競馬 |
| `src/providence/domain/enums.py` | オートレース（現状） |
| `src/providence/features/` | オートレース（現状） |
| `src/providence/scraper/` | オートレース（現状） |
| `src/providence/database/tables.py` | オートレース（現状） |
| `src/providence/model/` | 両方（共通基盤） |
| `src/providence/strategy/` | 両方（共通基盤） |
| `src/providence/probability/` | 両方（共通基盤） |
| `src/providence/backtest/` | 両方（共通基盤） |
| `src/providence/core/` | 両方（リファクタリング後） |

注: リファクタリング（Phase 0b）前は `src/providence/` 直下のモジュールがオートレース固有。
リファクタリング後は `autorace/` と `keiba/` に分離され、共通部分は `core/` に移動する。
