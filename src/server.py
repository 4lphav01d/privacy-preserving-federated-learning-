"""Flower server strategy: decompress client deltas, weighted FedAvg, comm metrics.

Per-round communication volumes are returned as aggregate-fit metrics (the strategy
may execute in a worker process, so results flow back through Flower's history,
not through in-memory mutation).
"""
from __future__ import annotations

import logging

import numpy as np
from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from . import compression as C

logger = logging.getLogger(__name__)


def _sum_metrics(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregate fit metrics: sum the byte totals, keep client count."""
    return {
        "uplink_bytes_sum": float(sum(m.get("uplink_bytes", 0) for _, m in metrics)),
        "num_clients": len(metrics),
    }


class CompressedFedAvg(FedAvg):
    """FedAvg over compressed deltas. Uplink: compressed payload bytes per client.
    Downlink: dense broadcast of the updated global model (design choice)."""

    def __init__(self, *args, compression: bool = False, rounds_file: str | None = None,
                 dataset: str = "mnist", seed: int = 0, **kwargs):
        kwargs.setdefault("fit_metrics_aggregation_fn", _sum_metrics)
        kwargs.setdefault("on_fit_config_fn", lambda r: {"server_round": r})
        kwargs.setdefault("on_evaluate_config_fn", lambda r: {"server_round": r})
        super().__init__(*args, **kwargs)
        self.compression = compression
        self.rounds_file = rounds_file
        from .models import build_model
        from .utils import flatten_state_dict, set_seeds
        set_seeds(seed)
        self._global = flatten_state_dict(build_model(dataset).state_dict())

    def initialize_parameters(self, client_manager=None):
        return ndarrays_to_parameters([self._global])

    def aggregate_fit(self, rnd, results: list[tuple[ClientProxy, FitRes]], failures):
        if not results:
            return None, {}
        weights = []
        uplink_bytes = 0
        results = sorted(results, key=lambda pair: pair[0].cid)  # deterministic summation order
        for _, fit_res in results:
            raw = C.unwrap_transport(fit_res.parameters.tensors[0])
            # true client-side payload size (transport may re-encode to float32)
            uplink_bytes += int(fit_res.metrics.get("uplink_bytes", 0))
            if self.compression:
                payload = C.deserialize_payload(raw)
                delta = C.decompress_delta(payload).numpy().ravel()
            else:
                delta = np.frombuffer(raw, dtype=np.float32)
            weights.append((delta, fit_res.num_examples))

        total = sum(n for _, n in weights)
        # float64 accumulation makes the sum order-robust at float32 precision
        avg = (np.sum([d.astype(np.float64) * n for d, n in weights], axis=0) / total).astype(np.float32)
        self._global = (self._global + avg).astype(np.float32)  # global + weighted avg delta
        flat_params = self._global
        downlink_bytes = flat_params.nbytes * len(results)

        logger.info("round %d: up=%.3fMB down=%.3fMB clients=%d",
                    rnd, uplink_bytes / 1e6, downlink_bytes / 1e6, len(results))
        if self.rounds_file:
            import json as _json
            with open(self.rounds_file, "a") as f:
                f.write(_json.dumps({"kind": "fit", "round": rnd,
                                     "uplink_bytes": int(uplink_bytes),
                                     "downlink_bytes": int(downlink_bytes),
                                     "num_clients": len(results)}) + "\n")
        return ndarrays_to_parameters([flat_params]), {
            "uplink_bytes_sum": float(uplink_bytes),
            "downlink_bytes_sum": float(downlink_bytes),
            "num_clients": float(len(results)),
        }
