"""Dataset acquisition, normalisation, and seed-locked Dirichlet non-IID partitioning."""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_ROOT = os.environ.get("FLDP_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _transforms(dataset: str, train: bool):
    mean, std = STATS[dataset]
    ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train and dataset == "cifar10":
        ops.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)


def load_datasets(dataset: str):
    cls = datasets.MNIST if dataset == "mnist" else datasets.CIFAR10
    train = cls(DATA_ROOT, train=True, download=True, transform=_transforms(dataset, True))
    test = cls(DATA_ROOT, train=False, download=True, transform=_transforms(dataset, False))
    return train, test


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    """Per-class Dirichlet split producing skewed per-client label mixtures."""
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx_c, cuts)):
            client_indices[k].extend(part.tolist())
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices


def get_partition(dataset: str, num_clients: int, alpha: float, seed: int, cache_dir: str):
    """Load cached partition or generate once, cache with content hash, return (shards, meta)."""
    os.makedirs(cache_dir, exist_ok=True)
    name = f"partition_{dataset}_K{num_clients}_a{alpha}_s{seed}.pkl"
    path = os.path.join(cache_dir, name)
    meta = {"dataset": dataset, "K": num_clients, "alpha": alpha, "seed": seed}

    if os.path.exists(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["meta"]["cache_hit"] = True
        return payload["shards"], payload["meta"]

    train, _ = load_datasets(dataset)
    labels = np.array(train.targets)
    shards = dirichlet_partition(labels, num_clients, alpha, seed)

    h = hashlib.sha256()
    for k in range(num_clients):
        h.update(np.asarray(sorted(shards[k]), dtype=np.int64).tobytes())
    meta.update({"hash": h.hexdigest()[:16], "sizes": [len(s) for s in shards], "cache_hit": False})

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"shards": shards, "meta": meta}, f)
    os.replace(tmp, path)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return shards, meta


def shard_loader(dataset: str, indices: list[int], batch_size: int, shuffle: bool,
                 drop_last: bool = False) -> DataLoader:
    train, _ = load_datasets(dataset)
    return DataLoader(Subset(train, indices), batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0)


def test_loader(dataset: str, batch_size: int = 256) -> DataLoader:
    _, test = load_datasets(dataset)
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
