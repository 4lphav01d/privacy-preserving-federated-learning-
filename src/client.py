"""Flower client: local DP-SGD training, delta computation, compressed uplink."""
from __future__ import annotations

import logging

import hashlib
import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient

from . import compression as C
from . import data as D
from .dp_engine import achieved_epsilon, attach_dp, dp_epoch, plain_epoch
from .utils import flatten_state_dict, set_seeds, unflatten_to_state_dict

logger = logging.getLogger(__name__)


class FlDpClient(NumPyClient):
    """One simulated client. The DP engine is attached once and persists across
    rounds so the whole training consumes a single (epsilon, delta) budget."""

    def __init__(self, cid: str, cfg: dict, shards, test_loader):
        self.cid = cid
        self.cfg = cfg
        self.indices = shards[int(cid)]
        self.test_loader = test_loader
        self.device = "cpu"  # DP-SGD hooks pinned to CPU per the device-routing rule

        from .models import build_model
        from .utils import set_seeds as _set_seeds
        _set_seeds(cfg["seed"] + 1000 + int(cid))   # deterministic init regardless of actor scheduling
        self.model = build_model(cfg["dataset"]).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
        self.criterion = nn.CrossEntropyLoss()
        self.engine = None
        self.train_loader = None
        self.k = int(cid)
        self._last_payload_bytes = 0
        self._eval_count = 0

    # -- Flower interface -------------------------------------------------
    def get_parameters(self, config=None):
        return [flatten_state_dict(self.model.state_dict())]

    def set_parameters(self, flat: np.ndarray):
        unflatten_to_state_dict(flat, self.model.state_dict())

    def fit(self, parameters, config):
        set_seeds(self.cfg["seed"] + 1000 + self.k)
        server_round = int(config.get("server_round", 0))
        self.set_parameters(parameters[0])
        n = len(self.indices)

        # dedicated attack-round capture: received global params + a fixed batch
        cap = self.cfg.get("attack_capture_round")
        if cap is not None and int(cap) == int(server_round) \
                and int(self.cid) == int(self.cfg.get("attack_capture_client", 0)):
            import torch as _t
            cap_loader = D.shard_loader(self.cfg["dataset"], self.indices[:512],
                                        batch_size=self.cfg.get("attack_images", 4) * 2,
                                        shuffle=False)
            bx, by = next(iter(cap_loader))
            _t.save({"params": parameters[0].copy(), "x": bx, "y": by},
                    self.cfg["attack_capture_path"])

        if self.cfg.get("target_epsilon"):
            if self.engine is None:
                total_epochs = self.cfg["rounds"] * self.cfg["local_epochs"]
                logical_loader = D.shard_loader(
                    self.cfg["dataset"], self.indices, self.cfg["logical_batch"],
                    shuffle=True, drop_last=True)
                self.engine, self.model, self.optimizer, self.train_loader = attach_dp(
                    self.model, self.optimizer, logical_loader,
                    total_epochs=total_epochs,
                    target_epsilon=float(self.cfg["target_epsilon"]),
                    delta=1.0 / n,
                )
            for _ in range(self.cfg["local_epochs"]):
                dp_epoch(self.model, self.train_loader, self.optimizer,
                         self.criterion, self.device, self.cfg["micro_batch"])
            eps_now = achieved_epsilon(self.engine, 1.0 / n)
        else:
            loader = D.shard_loader(self.cfg["dataset"], self.indices, self.cfg["logical_batch"], shuffle=True)
            for _ in range(self.cfg["local_epochs"]):
                plain_epoch(self.model, loader, self.optimizer, self.criterion, self.device)
            eps_now = None

        delta_vec = flatten_state_dict(self.model.state_dict()) - parameters[0]
        record = {"round": server_round, "cid": self.cid, "num_examples": n}
        if self.cfg.get("compression"):
            payload = C.compress_delta(torch.from_numpy(delta_vec), k=self.cfg["compression_k"])
            raw = C.serialize_payload(payload)
            uplink = np.frombuffer(raw, dtype=np.uint8)
        else:
            uplink = delta_vec.astype(np.float32)
        self._last_payload_bytes = uplink.nbytes
        record["uplink_bytes"] = int(uplink.nbytes)
        if eps_now is not None:
            record["epsilon_achieved"] = round(eps_now, 4)
        return [uplink], n, record

    def evaluate(self, parameters, config):
        import json as _json
        import os
        self.set_parameters(parameters[0])
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total
        path = self.cfg.get("_rounds_file")
        if path:
            with open(path, "a") as f:
                f.write(_json.dumps({"kind": "eval", "round": int(config.get("server_round", 0)),
                                     "cid": self.cid, "accuracy": acc,
                                     "num_examples": total}) + "\n")
        return 0.0, total, {"accuracy": acc, "cid": self.cid}
