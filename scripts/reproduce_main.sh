#!/usr/bin/env bash
# Reproduces Table 1, Table 4 (gradient norms) and Figure 1 (numpy, CPU; about 3-4 CPU-hours in total).
# Set JOBS to the number of parallel processes (default: number of performance cores / 2).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
JOBS=${JOBS:-4}
run_suite () {   # $1 = script, $2 = suite
  for k in $(seq 0 $((JOBS-1))); do python "$1" "$2" --shard "$k" "$JOBS" & done; wait
}
for s in grid hew hewx hewreg hewlag phew; do run_suite experiments/e2_grid.py "$s"; done
python experiments/figs_hew.py
python experiments/make_aistats_tables.py
echo "tables in paper_aistats/tables, figure in paper_aistats/figures"
