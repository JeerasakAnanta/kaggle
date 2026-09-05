#!/usr/bin/env bash
# Full pipeline: eda -> train -> predict
# Usage: bash run.sh [hgb|catboost|ensemble]  (default: ensemble)
set -e
cd "$(dirname "$0")"
PY="../../.venv/bin/python"
MODE="${1:-ensemble}"

mkdir -p data outputs/submissions
# point data/ at downloaded csvs if empty
for f in train.csv test.csv sample_submission.csv; do
  [ -f "data/$f" ] || [ ! -f "$f" ] || cp "$f" "data/$f"
done

$PY -m src.eda

case "$MODE" in
  hgb)      $PY -m src.train --model hgb --n-splits 5
            $PY -m src.predict --models hgb ;;
  catboost) $PY -m src.train --model catboost --n-splits 5
            $PY -m src.predict --models catboost ;;
  ensemble|*) $PY -m src.train --model hgb --n-splits 5
            $PY -m src.train --model catboost --n-splits 5
            $PY -m src.predict --models catboost hgb --weights 0.6 0.4 ;;
esac
echo "done -> outputs/submissions/submission.csv"
