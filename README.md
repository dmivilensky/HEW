# Code for "Heterogeneous-Horizon Exact-Weight Local SGD"

## Quick start
```bash
bash scripts/setup_mac.sh
bash scripts/smoke_test.sh
```
No manual download is needed: datasets are fetched by torchvision into `./data` on first use.

## Reproducing

```bash
bash scripts/run_torch_mac.sh
```

## Layout
* `fedweights/hew.py` — reference numpy implementation of all aggregation rules (debiased HEW with cross-round or
  sketched heterogeneity; RE / GLS / path rules; delayed or post-local timing; shrinkage; variance weights;
  fixed / decaying / SNR server steps; partial participation; SCAFFOLD; FedAvg).
* `fedweights/engine.py` — simulator used for the main table (baselines, plain HEW, ablations).
* `experiments/` — experiment suites (`e2_grid.py`, `e3_improvements.py`), table and figure scripts.
* `hew_torch/` — model-agnostic PyTorch implementation.
