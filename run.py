#!/usr/bin/env python
"""Single entry point: consumes a configuration YAML, executes a full training run."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("flwr").setLevel(logging.WARNING)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = {
        "mode": "train", "K": 4, "alpha": 0.5, "seed": 0,
        "lr": 0.1, "momentum": 0.9, "logical_batch": 64, "micro_batch": 8,
        "local_epochs": 1, "compression": False, "compression_k": 0.10,
        "target_epsilon": None,
    }
    defaults.update(cfg)
    return defaults


def _results_file() -> str:
    """Per-lane override so parallel lanes never append to the same file concurrently."""
    import os
    return os.environ.get("RESULTS_FILE", "results/results.jsonl")


def run_train(cfg: dict) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    import flwr as fl

    from src.client import FlDpClient
    from src.data import get_partition, test_loader
    from src.server import CompressedFedAvg
    from src.utils import ResultsLog, set_seeds

    set_seeds(cfg["seed"])
    shards, meta = get_partition(cfg["dataset"], cfg["K"], cfg["alpha"], cfg["seed"],
                                 cache_dir="results/artifacts")
    tl = test_loader(cfg["dataset"])

    import json as _json
    import os as _os
    run_id = _os.urandom(4).hex()
    rounds_file = f"results/rounds_{run_id}.jsonl"
    cfg["_rounds_file"] = rounds_file
    log_path = _results_file()
    log = ResultsLog(log_path, cfg, cfg["seed"])
    strategy = CompressedFedAvg(
        compression=bool(cfg["compression"]),
        rounds_file=rounds_file,
        dataset=cfg["dataset"], seed=cfg["seed"],
        fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=cfg["K"],
        min_evaluate_clients=cfg["K"], min_available_clients=cfg["K"],
    )

    cache = {}

    def client_fn(cid: str):
        if cid not in cache:
            cache[cid] = FlDpClient(cid, cfg, shards, tl)
        return cache[cid].to_client()

    with log as _:
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg["K"],
            config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
            strategy=strategy,
            client_resources={"num_cpus": float(os.cpu_count()), "num_gpus": 0.0},  # pool size 1: sequential, deterministic
        )

    fit_recs, eval_recs = {}, {}
    if _os.path.exists(rounds_file):
        with open(rounds_file) as f:
            for line in f:
                rec = _json.loads(line)
                (eval_recs if rec["kind"] == "eval" else fit_recs).setdefault(rec["round"], []).append(rec)
    for r in range(1, cfg["rounds"] + 1):
        f_recs = fit_recs.get(r, [])
        e_recs = eval_recs.get(r, [])
        uplink = sum(x["uplink_bytes"] for x in f_recs)
        downlink = sum(x["downlink_bytes"] for x in f_recs)
        acc = sum(x["accuracy"] * x["num_examples"] for x in e_recs) / max(sum(x["num_examples"] for x in e_recs), 1)
        log.log_round(
            round=r,
            accuracy=round(acc, 4),
            uplink_mb=round(uplink / 1e6, 3),
            downlink_mb=round(downlink / 1e6, 3),
            num_clients=f_recs[0]["num_clients"] if f_recs else 0,
        )
    if _os.path.exists(rounds_file):
        _os.remove(rounds_file)

    return log.finalize(
        status="ok",
        partition_hash=meta["hash"],
        partition_sizes=meta["sizes"],
        mode="train",
    )


def run_inversion(cfg: dict) -> dict:
    from src.evaluation import inversion_defence_experiment
    from src.utils import ResultsLog, set_seeds

    set_seeds(cfg["seed"])
    log = ResultsLog(_results_file(), cfg, cfg["seed"])
    rows = inversion_defence_experiment(cfg, [])
    return log.finalize(status="ok", mode="inversion", attack_rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="FedAvg + central-DP pipeline (client-side DP-SGD)")
    ap.add_argument("--config", required=True, help="path to a YAML experiment configuration")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[run] config={args.config} mode={cfg['mode']} dataset={cfg.get('dataset')} "
          f"rounds={cfg.get('rounds')} eps={cfg.get('target_epsilon')} compression={cfg.get('compression')}")
    result = run_inversion(cfg) if cfg["mode"] == "inversion" else run_train(cfg)
    print(f"[run] done: {result.get('status')} -> results/results.jsonl")
    sys.exit(0)


if __name__ == "__main__":
    main()
