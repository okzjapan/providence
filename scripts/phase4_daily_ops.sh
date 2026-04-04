#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_VALUE="${PROVIDENCE_OP_DATE:-$(date +%F)}"
TRACK_VALUE="${PROVIDENCE_OP_TRACK:-}"
RACE_VALUE="${PROVIDENCE_OP_RACE:-}"
JUDGMENT_TIME_VALUE="${PROVIDENCE_OP_JUDGMENT_TIME:-}"
MODEL_VERSION_VALUE="${PROVIDENCE_OP_MODEL_VERSION:-latest}"

usage() {
  cat <<'EOF'
Usage:
  scripts/phase4_daily_ops.sh scrape-day
  scripts/phase4_daily_ops.sh scrape-odds
  scripts/phase4_daily_ops.sh scrape-results
  scripts/phase4_daily_ops.sh predict
  scripts/phase4_daily_ops.sh report
  scripts/phase4_daily_ops.sh backtest

Environment variables:
  PROVIDENCE_OP_DATE            Target date (YYYY-MM-DD)
  PROVIDENCE_OP_TRACK           Track name, e.g. 川口
  PROVIDENCE_OP_RACE            Race number for predict
  PROVIDENCE_OP_JUDGMENT_TIME   ISO8601 for predict / HH:MM for backtest
  PROVIDENCE_OP_MODEL_VERSION   Model version (default: latest)
EOF
}

require_track() {
  if [[ -z "$TRACK_VALUE" ]]; then
    echo "PROVIDENCE_OP_TRACK is required" >&2
    exit 1
  fi
}

case "${1:-}" in
  scrape-day)
    require_track
    exec uv run providence scrape day --date "$DATE_VALUE" --track "$TRACK_VALUE"
    ;;
  scrape-odds)
    require_track
    exec uv run providence scrape odds --date "$DATE_VALUE" --track "$TRACK_VALUE"
    ;;
  scrape-results)
    require_track
    exec uv run providence scrape results --date "$DATE_VALUE" --track "$TRACK_VALUE"
    ;;
  predict)
    require_track
    if [[ -z "$RACE_VALUE" || -z "$JUDGMENT_TIME_VALUE" ]]; then
      echo "PROVIDENCE_OP_RACE and PROVIDENCE_OP_JUDGMENT_TIME are required for predict" >&2
      exit 1
    fi
    exec uv run providence predict \
      --date "$DATE_VALUE" \
      --track "$TRACK_VALUE" \
      --race "$RACE_VALUE" \
      --model-version "$MODEL_VERSION_VALUE" \
      --judgment-time "$JUDGMENT_TIME_VALUE" \
      --save
    ;;
  report)
    exec uv run providence report
    ;;
  backtest)
    if [[ -z "$JUDGMENT_TIME_VALUE" ]]; then
      echo "PROVIDENCE_OP_JUDGMENT_TIME is required for backtest" >&2
      exit 1
    fi
    TRACK_ARGS=()
    if [[ -n "$TRACK_VALUE" ]]; then
      TRACK_ARGS=(--track "$TRACK_VALUE")
    fi
    exec uv run providence backtest \
      --from "$DATE_VALUE" \
      --to "$DATE_VALUE" \
      --judgment-time "$JUDGMENT_TIME_VALUE" \
      --evaluation-mode fixed \
      --model-version "$MODEL_VERSION_VALUE" \
      "${TRACK_ARGS[@]}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
