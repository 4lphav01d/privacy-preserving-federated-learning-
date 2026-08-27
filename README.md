# Privacy-Preserving Federated Learning for Image Classification

This repository is the implementation pipeline for my final-year project, which studies the
privacy–utility–communication trade-off in federated learning (FL). The pipeline simulates
Flower-based federated training with client-side DP-SGD (Opacus) and top-k gradient
compression, then evaluates gradient inversion attacks (DLG, IG) against captured model updates.

## Research questions covered

- **RQ1/RQ2** — accuracy vs. privacy budget (ε) on MNIST and CIFAR-10 (non-IID Dirichlet partitioning)
- **RQ3** — communication cost and accuracy price of top-k compression, with and without DP
- **RQ4** — gradient inversion attack effectiveness before/after DP defence

## Repository layout

| Path | Description |
|---|---|
| `run.py` | Single entry point. Consumes a config YAML and executes a full run (training, attack capture, or inversion) |
| `verify.py` | Unit verification checkpoints (TC1–TC7). Run before any training |
| `analyse.py` | Chapter 5 analysis. Reads `results/results.jsonl`, outputs the summary/compression/inversion CSV tables and figures |
| `campaign.sh` | Sequential full-experiment campaign runner (resumable via `results/campaign_status.txt`) |
| `lane.sh` | Parallel-lane runner for the same campaign: `./lane.sh <lane> <results_file> <cfg.yaml>...` |
| `src/` | Core modules (see below) |
| `configs/` | 27 experiment configs: MNIST/CIFAR-10 ε grids, compression variants, attack captures, inversion, smoke test |
| `data/` | Downloaded datasets (MNIST, CIFAR-10). Auto-created on first use — not in git |
| `results/` | Measured outputs: `results.jsonl` (merged campaign records), CSV tables, logs, `artifacts/` (attack captures, partition pickles). Key result files are tracked; logs and binaries are not |
| `.venv/` | Python virtual environment — not in git |

### `src/` modules

| Module | Purpose |
|---|---|
| `models.py` | MNIST + CIFAR-10 CNN definitions (`build_model`, parameter counts) |
| `data.py` | Dataset download/normalisation and seed-locked Dirichlet non-IID partitioning |
| `client.py` | Flower client: local training with DP-SGD (Opacus), micro-batching, optional top-k compression |
| `server.py` | `CompressedFedAvg` strategy — cid-sorted aggregation, float64 accumulation, `global + avg(delta)` |
| `dp_engine.py` | DP-SGD engine plumbing (Opacus privacy accountant, per-sample gradients) |
| `compression.py` | Top-k gradient compression (uplink/downlink byte accounting) |
| `evaluation.py` | Gradient inversion attacks (DLG, IG) and the defence experiment |
| `utils.py` | Seeding, results logging (`ResultsLog` → JSONL) |

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/pip install flwr==1.34.0 opacus==1.6.0 torch==2.13.0 torchvision==0.28.0 numpy==2.4.6 ray==2.55.1 matplotlib pyyaml
```

## Datasets

MNIST and CIFAR-10 are downloaded automatically by torchvision on the first run
(`datasets.MNIST` / `datasets.CIFAR10` with `download=True` in `src/data.py`) — no manual
download is needed. They are stored in `data/`. To use a different location, set the
`FLDP_DATA` environment variable. Non-IID client partitions are generated with a
seed-locked Dirichlet split and cached with a content hash in `results/artifacts/` so
every run reuses identical shards.

## How to run

All commands use the venv python directly (`./.venv/bin/python`) and are run from this directory.

1. **Verify the installation** (should print PASS for TC1–TC7):

   ```bash
   ./.venv/bin/python verify.py
   ```

2. **Run a single experiment** (e.g. MNIST at ε=2):

   ```bash
   ./.venv/bin/python run.py --config configs/mnist_eps2.yaml
   ```

   Config `mode` selects the behaviour: `train` (default) trains a model,
   `mode: attack_capture_*` configs additionally dump an update capture to
   `results/artifacts/capture_{dp,nonprivate}.pt`, and `mode: inversion`
   replays DLG/IG against a capture and appends attack metrics.
   The smoke test is a 2-round sanity check: `configs/mnist_smoke.yaml`.

3. **Run the full campaign** (sequential, resumable — completed configs are skipped):

   ```bash
   ./campaign.sh
   ```

   Progress is logged to `results/campaign_status.txt`; per-run logs to
   `results/log_<name>.txt`.

4. **Run lanes in parallel** (how the campaign was actually executed):

   ```bash
   ./lane.sh A results/lane_A.jsonl configs/mnist_base.yaml configs/mnist_eps0.5.yaml ...
   ```

   Each lane appends to its own results file via the `RESULTS_FILE` env override so
   parallel lanes never write to `results/results.jsonl` concurrently. The `inversion`
   config within a lane waits for both capture files to exist before running.

5. **Build the analysis tables and figures** (after the campaign):

   ```bash
   ./.venv/bin/python analyse.py
   ```

   Outputs `results/summary_table.csv`, `results/compression_table.csv`,
   `results/inversion_table.csv` and the Chapter 5 figures.

### Gotchas

- **Clear `__pycache__/` after any code edit.** Ray actors have been observed running
  stale bytecode — this once caused a long determinism hunt:
  `find . -name __pycache__ -type d -exec rm -rf {} +`
- Keep the machine awake during long runs (`caffeinate -dims`) — sleep pauses training.
- CIFAR-10 with DP is lr-sensitive: `lr=0.1` diverges to chance; the tuned
  `configs/cifar_tuned_*.yaml` configs use `lr=0.02`. `analyse.py` reports both passes
  separately (rows are lr-aware).
- Flower legacy quirks (npy transport envelope, `on_fit_config_fn` round passing,
  never emitting `None` fit metrics) are handled in `src/` — do not "simplify" them away.

## Results provenance

All numbers reported in the thesis chapters are measured, stored in
`results/results.jsonl` (123 records), and summarised by `analyse.py` into the CSV
tables. A sequential spot-check re-run of `mnist_comp_base` reproduced the lane result
bit-for-bit (documented in Chapter 4 §4.5).
