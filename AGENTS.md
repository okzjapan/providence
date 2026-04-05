# Providence プロジェクト指示書

## プロジェクト概要

オートレースおよび JRA 中央競馬の AI 予想システム。LightGBM LambdaRank + Plackett-Luce + Kelly 最適化のスタックを共有基盤として、複数の公営競技の予測を行う。

## 自動コンテキスト読み込みルール

**コードを変更する前に、対象ドメインのドキュメントとコードを必ず読み込むこと。**

### ドメイン判定（パスベース）

以下のパスパターンに該当するファイルを変更・作成する場合、対応するドメインスキルを適用する。

**オートレース** → `.claude/skills/autorace-domain/SKILL.md` の手順に従う:
- `src/providence/domain/`
- `src/providence/features/`
- `src/providence/scraper/`
- `src/providence/database/`
- `src/providence/autorace/`（リファクタリング後）
- `tests/` 内で上記に対応するテスト

**競馬（JRA）** → `.claude/skills/keiba-domain/SKILL.md` の手順に従う:
- `src/providence/keiba/`
- `tests/keiba/`

**共通基盤**（両ドメインに影響） → 両方のスキルの手順に従う:
- `src/providence/model/`
- `src/providence/strategy/`
- `src/providence/probability/`
- `src/providence/backtest/`
- `src/providence/core/`（リファクタリング後）

### 判定できない場合

ファイルパスだけでは判定できない場合は、変更内容の意味から判断する。
「オートレース」「選手」「試走」「車券」→ オートレースドメイン。
「競馬」「馬」「騎手」「芝」「ダート」「血統」「馬券」→ 競馬ドメイン。

## ドキュメント同期ルール

**コードを変更したら、対応するドメインドキュメントを必ず同時に更新すること。**

このルールはプロジェクト全体で最も重要な規約であり、例外は認めない。
詳細なマッピングは `.claude/skills/domain-doc-sync/SKILL.md` を参照。

## 技術スタック

- Python 3.12+, uv（パッケージ管理）
- LightGBM, Polars, SQLAlchemy, Alembic, Typer
- リンター: ruff（line-length=120, select=E,F,I,W）
- テスト: pytest + pytest-asyncio

## コーディング規約

- 日本語コメントは許可するが、変数名・関数名・クラス名は英語
- 変数名は `docs/{autorace,keiba}/glossary.md` に準拠
