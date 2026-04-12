# オートレース自動投票デーモンのセットアップ

別の macOS 環境でオートレース自動投票デーモンを動かすためのセットアップ手順。

## 前提条件

- macOS
- Python 3.12+
- uv（パッケージマネージャ）
- Google Chrome

## 手順

### 1. 手動コピーが必要なファイル

以下のファイルは git に含まれないため、元の環境から手動でコピーすること。

| ファイル/ディレクトリ | サイズ | 内容 |
|---|---|---|
| `.env` | 1KB | Slack Webhook URL, WinTicket PIN 等 |
| `data/models/v013/` | 22MB | 学習済みモデル（6モデル + Model B + キャリブレータ + 閾値） |
| `data/providence.db` | 3.3GB | レース履歴・オッズ・結果の全データ |

転送コマンド例（元の Mac から）:
```bash
rsync -avz .env data/models/ data/providence.db user@new-mac:~/dev/providence/
```

### 2. 依存関係のインストール

```bash
cd ~/dev/providence
uv sync
uv add --dev playwright
uv run python -m playwright install chromium
```

### 3. 動作確認

```bash
# DB とモデルの存在確認
ls data/providence.db data/models/v013/metadata.json data/models/v013/value_model_b.txt

# .env の設定確認
uv run python -c "from providence.config import get_settings; s = get_settings(); print(f'Slack: {bool(s.slack_webhook_url)}, PIN: {bool(s.winticket_pin)}')"

# 簡易テスト（Slack 通知が届くか）
uv run providence autobet tick --force-race 川口 --force-race-number 1 --dry-run --no-save
```

### 4. WinTicket 用 Chrome の起動

通常の Chrome とは別に、WinTicket 専用の Chrome を起動する。

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.chrome-winticket \
  https://www.winticket.jp/
```

ブラウザで WinTicket にログイン（2FA 突破）する。

### 5. デーモン起動

```bash
# 通知 + 自動投票
cd ~/dev/providence && uv run providence autobet daemon --auto-bet
```

### 6. 停止方法

- `Ctrl+C`: 安全に停止
- `pkill -f "providence autobet daemon"`: 別ターミナルから停止
- `touch data/KILL_SWITCH`: 投票のみ緊急停止（デーモンは継続）

### デーモンが行うこと

- 毎日 09:00 JST: morning-sync（全場スクレイプ + デイリーサマリー Slack 通知）
- 09:00-23:00 JST: 3分間隔で tick
  - Phase 1 (T-10min): 試走タイム + 予測 + T-10min オッズ保存
  - Phase 2 (T-7min): 最新オッズ + 戦略 + Slack 通知 + S ランクなら自動投票
  - 後処理 (T+60min): 確定オッズ + レース結果 + 損益照合
- 23:00-09:00 JST: スリープ（翌朝を待つ）

### 投票戦略（実証済み）

- 券種: 単勝 + ワイド
- kelly: 0.05（厳守）
- 信頼度: >= 0.90
- EV: >= 0.40
- S ランクのみ自動投票
- バックテスト ROI: +54.5%
