# Providence - オートレース予測システム

オートレースの予測モデルを構築し、Kelly基準に基づく最適な車券購入推薦を行うシステム。

## セットアップ

```bash
uv sync                    # 依存関係インストール
uv sync --extra dev        # 開発ツール (ruff, pytest等)
uv sync --extra notebook   # Jupyter (EDA時)
cp .env.example .env       # 設定ファイル
uv run alembic upgrade head
providence db stats        # DB初期化確認
```

Phase 3 以降は、旧 `odds_snapshot` に入っていた払戻データを新しい `ticket_payouts` に移すため、既存 DB に対して一度だけ次を実行する:

```bash
providence db backfill-payouts
```

## 基本コマンド

```bash
providence scrape today                      # 本日の開催情報
providence scrape players                    # 選手マスタ収集
providence scrape day --date 2026-03-29      # 特定日データ収集
providence scrape odds --date 2026-03-29 --track 川口 --race 12  # 市場オッズ取得
providence scrape historical --from 2024-01-01 --to 2024-12-31 --resume  # 過去データ
providence predict --date 2026-03-29 --track 川口 --race 12 --bankroll 10000
providence backtest --from 2026-03-01 --to 2026-03-31 --judgment-time 10:00 --evaluation-mode fixed --model-version v003
providence report
providence report --refresh
providence retrain --compare-window-days 28
```

## Phase 3 運用メモ

- `odds_snapshot` は Phase 3 から `市場オッズ専用`。`ticket_payouts` が払戻専用。
- 収益バックテストは `ingestion_batch_id` 付きの市場オッズがある期間だけを対象にする。
- `predict` / `backtest` は `judgment_time` 以前の市場オッズ batch のみを使う。
- 旧履歴は `db backfill-payouts` 実行後、精算用データとして `ticket_payouts` から参照される。

## Phase 4 運用フロー

Phase 4 は、まず `ローカル単一マシン + cron` を前提に運用する。
ただしジョブ境界は CLI コマンドで閉じているため、将来的には GitHub Actions などの CI の `run:` ステップへ移しやすい。

典型的な日次フロー:

```bash
uv run providence scrape day --date 2026-04-02 --track 伊勢崎
uv run providence scrape odds --date 2026-04-02 --track 伊勢崎
uv run providence predict --date 2026-04-02 --track 伊勢崎 --race 12 --judgment-time 2026-04-02T10:00:00 --save
uv run providence report
```

ラッパースクリプト:

```bash
cp scripts/phase4_env.example .env.phase4
source .env.phase4
scripts/phase4_daily_ops.sh scrape-day
scripts/phase4_daily_ops.sh scrape-odds
scripts/phase4_daily_ops.sh predict
scripts/phase4_daily_ops.sh report
```

cron 例:

```cron
0 7 * * * cd /path/to/providence && source .env.phase4 && scripts/phase4_daily_ops.sh scrape-day >> data/logs/phase4.log 2>&1
0 10 * * * cd /path/to/providence && source .env.phase4 && scripts/phase4_daily_ops.sh scrape-odds >> data/logs/phase4.log 2>&1
5 10 * * * cd /path/to/providence && source .env.phase4 && scripts/phase4_daily_ops.sh predict >> data/logs/phase4.log 2>&1
0 22 * * * cd /path/to/providence && source .env.phase4 && scripts/phase4_daily_ops.sh report >> data/logs/phase4.log 2>&1
```

運用上の注意:

- `report` は Phase 4 では最小の運用確認コマンドであり、`scrape_log`, `strategy_runs`, `predictions`, `latest model` の確認を目的にする。
- `retrain` や本格的な成績追跡は Phase 5 の責務。
- CI に移行する場合も、まずはこの README のコマンド列をそのままジョブ化する想定にする。

## Phase 5 フィードバックループ

Phase 5 では `report --refresh` が紙トレードベースのフィードバック更新入口になる。

```bash
uv run providence report --refresh
```

この処理では次を順に行う:

- 直近の `StrategyRun` / `Prediction` と `TicketPayout` を突合して紙トレード損益を更新
- `ModelPerformance` を `1w / 4w / 12w` ウィンドウで upsert
- ドリフト判定を行い、`report` に警告を表示

候補モデルの再学習:

```bash
uv run providence retrain --compare-window-days 28
uv run providence retrain --compare-window-days 28 --promote
```

運用上の注意:

- `retrain` はデフォルトでは `latest` を更新しない。候補モデルは保存されるが、本番昇格は `--promote` 明示時のみ。
- `report --refresh` の freshness は `BettingLog` と `ModelPerformance` の更新時刻を基準に表示する。
- `ModelPerformance` は本番/紙トレード由来の集計のみを保存し、backtest 結果とは混ぜない。
