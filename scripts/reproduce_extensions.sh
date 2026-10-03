#!/usr/bin/env bash
# Reproduces Tables 2, 5-7 (sketches, variance weights, server steps, partial participation, SCAFFOLD, 5 seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
JOBS=${JOBS:-4}
for s in sketch varw step partial scaffold; do
  for k in $(seq 0 $((JOBS-1))); do python experiments/e3_improvements.py "$s" --shard "$k" "$JOBS" & done; wait
done
for k in $(seq 0 $((JOBS-1))); do python experiments/e3_improvements.py seeds --seeds 0 1 2 3 4 --shard "$k" "$JOBS" & done; wait
python experiments/make_aistats_tables.py
