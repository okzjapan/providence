# keiba-domain リファレンスガイド

このスキルの詳細な背景知識は `docs/keiba/` ディレクトリに体系的に整理されている。
必要に応じて以下のドキュメントを参照すること。

## ドキュメントマップ

| 参照タイミング | ドキュメント |
|---------------|-------------|
| DB設計・Enum定義 | `docs/keiba/01-jra-structure.md` |
| 特徴量の追加・修正 | `docs/keiba/02-feature-catalog.md` |
| パーサー実装・データパイプライン | `docs/keiba/03-data-sources.md` |
| モデル学習・バリデーション | `docs/keiba/04-model-design.md` |
| Kelly最適化・券種選定 | `docs/keiba/05-strategy-and-market.md` |
| コードレビュー・品質チェック | `docs/keiba/06-pitfalls.md` |
| 命名規則・変数名の確認 | `docs/keiba/glossary.md` |

## 重要な数値定数

| 定数 | 値 | 出典 |
|------|-----|------|
| JRA競馬場数 | 10場 | 01-jra-structure.md §3 |
| 最大出走頭数 | 18頭 | 01-jra-structure.md §2 |
| 3連単最大組み合わせ | 4,896通り | 01-jra-structure.md §2 |
| 単勝控除率 | 20% | glossary.md 券種表 |
| 3連単控除率 | 27.5% | glossary.md 券種表 |
| JRA市場キャリブレーション誤差 | ECE=0.0025 | 05-strategy-and-market.md §1 |
| 斤量1kgの影響 | ≒0.084秒≒0.42馬身 | 01-jra-structure.md §6 |
| 馬体重比閾値 | 12.5% | 01-jra-structure.md §6 |
