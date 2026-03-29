# Providence - オートレース予測システム

オートレースの予測モデルを構築し、Kelly基準に基づく最適な車券購入推薦を行うシステム。

## セットアップ

```bash
uv sync                    # 依存関係インストール
uv sync --extra dev        # 開発ツール (ruff, pytest等)
uv sync --extra notebook   # Jupyter (EDA時)
cp .env.example .env       # 設定ファイル
providence db stats        # DB初期化確認
```

## 基本コマンド

```bash
providence scrape today                      # 本日の開催情報
providence scrape players                    # 選手マスタ収集
providence scrape day --date 2026-03-29      # 特定日データ収集
providence scrape historical --from 2024-01-01 --to 2024-12-31 --resume  # 過去データ
```
