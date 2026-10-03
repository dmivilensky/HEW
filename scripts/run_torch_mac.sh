#!/usr/bin/env bash
# All neural-network experiments, in order of priority for the paper. PyTorch on Apple Metal (MPS), CUDA or CPU.
# Datasets are downloaded automatically. Results: results/torch/<suite>_<setting>/<method>_s<seed>.json;
# summary: results/torch/summary.txt; figures and LaTeX tables: results/torch/figures (refreshed after every block).
#
# Blocks (run in this order; PRIORITY selects a subset, e.g. PRIORITY="1 2"):
#  1  hn    noise-heterogeneous systems (MLP, b_i in {2,...,256}, lr 0.3), IID / Dir(1) / Dir(0.3)   main result
#  2  main  moderate systems heterogeneity (CNN, b_i in {8,...,128}, lr 0.1), IID / Dir(1) / Dir(0.3)
#  3  a01   strong data heterogeneity Dir(0.1) for both suites hn and main (completes their grids)
#  4  pp    partial participation: hn systems, 50 clients, 30% per round, IID / Dir(0.3)
#  5  strag stragglers (horizons re-drawn every round): hn systems, IID / Dir(0.3)
#  6  abl   ablations on hn, Dir(1): post-local weights, SNR server step
#  7  seeds seeds 3 and 4 for blocks 1 and 2
#  8  cifar CIFAR-10 / ResNet-18, 50 clients, 20% participation, Dir(0.3) (slow; only with CIFAR=1)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
ROUNDS=${ROUNDS:-2000}
SEEDS=${SEEDS:-"0 1 2"}
PRIORITY=${PRIORITY:-"1 2 3 4 5 6 8"}
# p: normalized averaging (FedNova-type); fedavg: unnormalized; iv: inverse-variance; scaffold: control variates;
# hew_lm: debiased HEW, theta = lr * lambda_max (main variant); hew: online theta; hew_t4: theta = 4;
# hew_sk: gradient-sketch heterogeneity; scaffold_hew: control variates + debiased HEW weights
METHODS=${METHODS:-"p fedavg iv scaffold hew_lm hew hew_t4 hew_sk scaffold_hew"}
ABL_METHODS="p hew_lm hew_post p_adapt hew_adapt"
HN="--model mlp --lr 0.3 --b_set 2 4 8 64 256"
MAIN="--model cnn --lr 0.1 --b_set 8 16 32 64 128"
alpha_of() { case "$1" in iid) echo inf;; a1) echo 1.0;; a03) echo 0.3;; a01) echo 0.1;; *) echo "unknown setting $1" >&2; exit 1;; esac; }

# block <suite-name> "<settings>" "<seeds>" "<methods>" <common flags...>
block() {
  local name=$1 settings=$2 seeds=$3 methods=$4; shift 4
  for setting in $settings; do
    local alpha; alpha=$(alpha_of "$setting")
    for seed in $seeds; do for m in $methods; do
      python -m hew_torch.run --dataset fmnist --alpha "$alpha" --rounds "$ROUNDS" --eval_every 5 \
        --method "$m" --seed "$seed" --skip_existing "$@" \
        --out "results/torch/fmnist_${name}_${setting}/${m}_s${seed}.json"
    done; done
  done
  python experiments/plot_torch.py results/torch > /dev/null || true
  python experiments/summarize_torch.py results/torch > results/torch/summary.txt || true
}

mkdir -p results/torch
for P in $PRIORITY; do
  echo "=== block $P ===" >&2
  case "$P" in
    1) block mlp_hn   "iid a1 a03" "$SEEDS" "$METHODS" --n 20 $HN ;;
    2) block cnn_main "iid a1 a03" "$SEEDS" "$METHODS" --n 20 $MAIN ;;
    3) block mlp_hn   "a01" "$SEEDS" "$METHODS" --n 20 $HN
       block cnn_main "a01" "$SEEDS" "$METHODS" --n 20 $MAIN ;;
    4) block mlp_pp    "iid a03" "$SEEDS" "$METHODS" --n 50 --participation 0.3 $HN ;;
    5) block mlp_strag "iid a03" "$SEEDS" "$METHODS" --n 20 --stragglers $HN ;;
    6) block mlp_abl   "a1" "$SEEDS" "$ABL_METHODS" --n 20 $HN ;;
    7) block mlp_hn   "iid a1 a03 a01" "3 4" "$METHODS" --n 20 $HN
       block cnn_main "iid a1 a03 a01" "3 4" "$METHODS" --n 20 $MAIN ;;
    8) if [ "${CIFAR:-0}" = "1" ]; then
         for seed in $SEEDS; do for m in $METHODS; do
           python -m hew_torch.run --dataset cifar10 --model resnet18 --n 50 --participation 0.2 --alpha 0.3 \
             --rounds "$ROUNDS" --lr 0.2 --wd 5e-4 --eval_every 10 --method "$m" --seed "$seed" --skip_existing \
             --out "results/torch/c10_r18_a03/${m}_s${seed}.json"
         done; done
       fi ;;
    *) echo "unknown block $P" >&2; exit 1 ;;
  esac
done

python experiments/summarize_torch.py results/torch | tee results/torch/summary.txt
python experiments/plot_torch.py results/torch
