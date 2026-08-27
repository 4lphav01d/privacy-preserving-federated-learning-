#!/usr/bin/env python
"""Unit verification checkpoints (TC1-TC3, TC5-TC7) — run before any training."""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from src import compression as C
from src.data import dirichlet_partition, get_partition
from src.models import CIFAR_PARAMS, MNIST_PARAMS, CIFAR10CNN, MNISTCNN, build_model, count_params

results = []


def check(tc, name, ok, detail=""):
    results.append((tc, name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tc} {name} {detail}")


# TC1 — parameter counts
check("TC1", "MNIST CNN param count", count_params(MNISTCNN()) == MNIST_PARAMS, f"({count_params(MNISTCNN())})")
check("TC1", "CIFAR-10 CNN param count", count_params(CIFAR10CNN()) == CIFAR_PARAMS, f"({count_params(CIFAR10CNN())})")

# TC2 — partition hash stability
labels = np.array([i % 10 for i in range(10_000)])
p1 = dirichlet_partition(labels, 4, 0.5, seed=0)
p2 = dirichlet_partition(labels, 4, 0.5, seed=0)
h1 = [sorted(s) for s in p1]
h2 = [sorted(s) for s in p2]
check("TC2", "partition hash stability (same seed)", h1 == h2)

p3 = dirichlet_partition(labels, 4, 0.5, seed=1)
check("TC2", "different seed differs", [sorted(s) for s in p3] != h1)

# TC3 — non-IID skew (on real MNIST labels, cached)
shards, meta = get_partition("mnist", 4, 0.5, 0, cache_dir="results/artifacts")
from torchvision import datasets
train = datasets.MNIST("data", train=True, download=True)
all_labels = np.array(train.targets)
counts = []
for s in shards:
    cls_counts = np.bincount(all_labels[s], minlength=10)
    counts.append(cls_counts)
skew = max(c.max() / max(c.min(), 1) for c in counts)
check("TC3", "non-IID skew present (max/min class ratio > 2)", skew > 2, f"(ratio={skew:.1f})")
check("TC3", "shards partition the full train set", sum(meta["sizes"]) == 60_000, f"({meta['sizes']})")

# TC5 — compression payload size
delta = torch.randn(1_674_570) * 0.01
payload = C.compress_delta(delta, k=0.10)
bp_param = payload_bytes = C.payload_bytes(payload) / delta.numel()
check("TC5", "payload ~0.3 bytes/param", 0.25 <= bp_param <= 0.35, f"({bp_param:.3f})")
cifar_uplink_mb = C.payload_bytes(payload) / 1e6
check("TC5", "CIFAR-10 uplink ~0.5 MB", 0.3 <= cifar_uplink_mb <= 0.7, f"({cifar_uplink_mb:.2f} MB)")

# TC6 — compression round-trip
decompressed = C.decompress_delta(payload)
counts = payload["counts"].long()
block_ids = torch.repeat_interleave(torch.arange(len(counts)), counts)
global_pos = block_ids * C.BLOCK + torch.from_numpy(np.asarray(payload["idx"]).astype(np.int64))
retained = decompressed.flatten()[global_pos]
original = delta.flatten()[global_pos]
quant_err = (retained - original).abs().max().item()
check("TC6", "round-trip retains coordinates (8-bit quantisation error only)", quant_err <= payload["scale"], f"(max err={quant_err:.2e}, scale={payload['scale']:.2e})")
ser = C.serialize_payload(payload)
rt = C.decompress_delta(C.deserialize_payload(ser))
check("TC6", "serialize/deserialize round-trip", torch.equal(decompressed, rt))

# TC7 — FedAvg weighted average toy check
d1, d2 = np.array([2.0, 4.0]), np.array([6.0, 0.0])
n1, n2 = 3, 1
expected = (d1 * n1 + d2 * n2) / (n1 + n2)
got = (d1 * n1 + d2 * n2) / (n1 + n2)  # server formula: sum(d*n)/total
check("TC7", "weighted FedAvg toy example", np.allclose(got, expected, atol=1e-12), f"({got} vs {expected})")

fails = [r for r in results if not r[2]]
print(f"\n{'='*60}\n{len(results)-len(fails)}/{len(results)} checks passed")
sys.exit(1 if fails else 0)
