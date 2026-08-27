"""Shared utilities: seed locking, JSONL results log, hardware tag, flatten helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_tag() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ResultsLog:
    """Append-only JSONL results log (one record per run, per-round entries inside)."""

    def __init__(self, path: str, config: dict, seed: int):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "seed": seed,
            "hardware": hardware_tag(),
            "rounds": [],
        }

    def log_round(self, **kwargs) -> None:
        self.base["rounds"].append(kwargs)

    def finalize(self, **kwargs) -> dict:
        self.base.update(kwargs)
        self.base["wall_clock_s"] = round(time.time() - self._t0, 1) if hasattr(self, "_t0") else None
        with open(self.path, "a") as f:
            f.write(json.dumps(self.base) + "\n")
        return self.base

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        if not self.base.get("finalized"):
            self.finalize(status="error")


def flatten_state_dict(state_dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state_dict.values()]).astype(np.float32)


def unflatten_to_state_dict(flat: np.ndarray, state_dict) -> None:
    offset = 0
    for k, v in state_dict.items():
        n = v.numel()
        v.data.copy_(torch.from_numpy(flat[offset:offset + n].reshape(v.shape)))
        offset += n
    assert offset == flat.size
