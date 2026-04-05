# autorace-domain リファレンスガイド

このスキルの詳細な背景知識は `docs/autorace/` ディレクトリに体系的に整理されている。
必要に応じて以下のドキュメントを参照すること。

## ドキュメントマップ

| 参照タイミング | ドキュメント |
|---------------|-------------|
| DB設計・Enum定義 | `docs/autorace/01-autorace-structure.md` |
| 特徴量の追加・修正 | `docs/autorace/02-feature-catalog.md` |
| スクレイパー実装・データパイプライン | `docs/autorace/03-data-sources.md` |
| モデル学習・バリデーション | `docs/autorace/04-model-design.md` |
| Kelly最適化・車券選定 | `docs/autorace/05-strategy-and-market.md` |
| コードレビュー・品質チェック | `docs/autorace/06-pitfalls.md` |
| 命名規則・変数名の確認 | `docs/autorace/glossary.md` |

## 重要な数値定数

| 定数 | 値 | 出典 |
|------|-----|------|
| レース場数 | 5場 | 01-autorace-structure.md §2 |
| 最大出走数 | 8車 | 01-autorace-structure.md §3 |
| レース距離 | 3,100m | 01-autorace-structure.md §3 |
| 控除率 | 全券種25% | 01-autorace-structure.md §5 |
| 3連単組み合わせ数 | 336通り | 04-model-design.md §3 |
| ハンデ範囲 | 0-110m | 01-autorace-structure.md §9 |
| 信頼度の重み | spread 0.6 + coverage 0.4 | 04-model-design.md §6 |
| fractional_kelly 既定値 | 0.25 | 05-strategy-and-market.md §3 |
| min_confidence 既定値 | 0.1 | 05-strategy-and-market.md §4 |
| max_total_stake 既定値 | 10,000円 | 05-strategy-and-market.md §3 |
