# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

Implementation pipeline for a final-year research project on the privacy–utility–communication
trade-off in federated learning: FedAvg over Flower with client-side DP-SGD (Opacus),
top-k gradient compression, and gradient inversion attack evaluation (DLG, IG) on MNIST/CIFAR-10.
Human-facing documentation lives in `README.md` — read it first for context on the file layout
and how to run experiments. The measured results in `results/` are thesis evidence.

## Environment

- Python 3.11 virtual environment at `.venv/`. Always invoke it explicitly as
  `./.venv/bin/python` — never assume an activated venv or bare `python`.
- Key pinned versions: `flwr==1.34.0`, `opacus==1.6.0`, `torch==2.13.0`,
  `torchvision==0.28.0`, `numpy==2.4.6`, `ray==2.55.1`. Do not add or upgrade dependencies
  without asking; deterministic behaviour depends on them.
- Datasets auto-download via torchvision into `data/` on first use (`FLDP_DATA` env var to
  relocate). Never commit `data/`, `.venv/`, logs, or binary artifacts — `.gitignore` covers it.
- macOS: use `caffeinate -dims` around long training runs (sleep pauses the simulation).

## Commands

Run these from the repository root.

```bash
# Verify installation and unit checkpoints TC1-TC7 — run before any training work
./.venv/bin/python verify.py            # must print PASS for every check

# Quick end-to-end sanity run (~2 rounds)
./.venv/bin/python run.py --config configs/mnist_smoke.yaml

# Full single experiment (any configs/*.yaml)
./.venv/bin/python run.py --config configs/mnist_eps2.yaml

# Analysis tables + figures from results/results.jsonl
./.venv/bin/python analyse.py
```

There is no linter/test-runner config; `verify.py` plus a smoke run is the verification
workflow. Full campaign runs go through `./campaign.sh` (sequential) or `./lane.sh`
(parallel lanes) — do not improvise alternatives without asking.

## Critical invariants — do not break

1. **Clear `__pycache__/` after ANY code edit**, then re-run:
   `find . -name __pycache__ -type d -exec rm -rf {} +`
   Ray actors have executed stale bytecode after edits before, causing a long determinism hunt.

2. **Determinism contract.** Runs are bit-for-bit reproducible given the same code:
   fixed seeds everywhere, Ray pool size 1 (sequential clients), cid-sorted aggregation,
   float64 accumulation, update rule `global + avg(delta)`. Do not change any of these.
   Reproducibility claims in the thesis rest on it.

3. **Flower legacy quirks are intentional** — do not "simplify" them away:
   - `unwrap_transport` handles Flower's npy transport envelope
   - `on_fit_config_fn` passes `server_round` to fit instructions
   - fit metrics must never be `None`
   - parallel lanes write via the `RESULTS_FILE` env override so they never append to
     `results/results.jsonl` concurrently

4. **Results are append-only evidence.** `results/results.jsonl` holds all measured campaign
   records. Never hand-edit, delete, or "clean up" existing records — only pipeline runs may
   append. `analyse.py` reads this file; CIFAR rows are lr-aware (the diverged lr=0.1 pass and
   tuned lr=0.02 pass are reported separately).

5. **CIFAR-10 DP training is lr-sensitive**: lr=0.1 diverges to chance (~10%); tuned configs
   use lr=0.02. Preserve both passes in analysis output.

## Code conventions

- Match existing style: `from __future__ import annotations` at top of every module,
  type hints on public functions, concise docstrings on entry points, no inline comments
  unless non-obvious behaviour demands one.
- Configs are complete YAML snapshots of a run's parameters (one file per experiment under
  `configs/`) — keep new experiments in that format rather than CLI flags.
- Results/logging plumbing goes through `src/utils.py::ResultsLog`; byte accounting through
  `src/compression.py`.

## Git etiquette

- Author identity comes from the user's global git config — do not override.
- Use meaningful, descriptive commit messages (imperative mood), split by concern
  (e.g. core / configs / tooling / results / docs).
- Never commit unless explicitly asked.
