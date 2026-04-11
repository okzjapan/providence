# オートレース予想 AI ドメイン知識

オートレース予想システムのドメイン知識を体系的に整理したドキュメント群。
実装中の意思決定基盤として、また AI アシスタントへのコンテキスト注入元として利用する。

## ドキュメントマップ

| ファイル | 内容 | 主な参照タイミング |
|----------|------|-------------------|
| [01-autorace-structure.md](01-autorace-structure.md) | オートレースの競技構造リファレンス | DB 設計、Enum 定義、特徴量設計 |
| [02-feature-catalog.md](02-feature-catalog.md) | 特徴量カタログ（全カテゴリ・実装済み特徴量の定義） | 特徴量エンジニアリング、リーク検証 |
| [03-data-sources.md](03-data-sources.md) | データソース（autorace.jp / oddspark）仕様 | パーサー実装、データパイプライン |
| [04-model-design.md](04-model-design.md) | モデル設計指針 | モデル学習、バリデーション |
| [05-strategy-and-market.md](05-strategy-and-market.md) | 車券戦略・Kelly 最適化 | 戦略パラメータ調整 |
| [06-pitfalls.md](06-pitfalls.md) | 陥りやすい罠と対策チェックリスト | 全フェーズ（レビュー時） |
| [07-strategy-search-log.md](07-strategy-search-log.md) | 戦略パラメータ探索ログ・採用設定 | 戦略最適化、モデル比較 |
| [glossary.md](glossary.md) | 用語集（日本語 / 英語 / コード変数名） | 命名規則、コードレビュー |

## 運用ルール

- **リビングドキュメント**: 新しい知見や実装上の発見があれば随時更新する
- **特徴量カタログ**: 新しい特徴量を追加したら `02-feature-catalog.md` を必ず同時に更新する
- **用語集**: 新しいドメイン用語をコードに導入したら `glossary.md` に追記する
