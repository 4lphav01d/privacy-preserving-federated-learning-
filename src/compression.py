"""Top-k sparsification + 8-bit uniform quantisation with 16-bit block-local indices.

The delta is divided into fixed blocks of 65,536 parameters so every retained
element's position fits an unsigned-range 16-bit local offset. Block membership
is implied by a per-block counts array (4 bytes per block, ~26 blocks for the
CIFAR-10 model), so the payload stays at ~0.3 bytes per retained-parameter
envelope: 0.1 x (1 value byte + 2 index bytes).
"""
from __future__ import annotations

import numpy as np
import torch

BLOCK = 1 << 16  # 65,536 parameters per block


def compress_delta(delta: torch.Tensor, k: float = 0.10) -> dict:
    flat = delta.detach().cpu().flatten()
    n_keep = max(1, int(flat.numel() * k))
    thresh = torch.kthvalue(flat.abs().float(), flat.numel() - n_keep + 1).values
    mask = flat.abs() >= thresh
    vals = flat[mask].float()
    scale = (vals.abs().max() / 127.0).clamp(min=1e-12)
    q = torch.round(vals / scale).clamp(-127, 127).to(torch.int8)

    pos = mask.nonzero().squeeze(1)                 # global positions (int64)
    block_ids = pos // BLOCK
    local_idx = (pos % BLOCK).numpy().astype(np.uint16)  # unsigned 16-bit: 0..65,535
    n_blocks = int(flat.numel() // BLOCK + (1 if flat.numel() % BLOCK else 0))
    counts = torch.bincount(block_ids, minlength=n_blocks).to(torch.int32)

    return {"shape": tuple(delta.shape), "scale": float(scale.item()),
            "q": q, "idx": local_idx, "counts": counts}


def decompress_delta(payload: dict, device: str = "cpu") -> torch.Tensor:
    q = torch.as_tensor(payload["q"]).long()
    idx = np.asarray(payload["idx"]).astype(np.int64)
    counts = torch.as_tensor(payload["counts"]).long()
    offsets = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), counts)
    pos = offsets * BLOCK + torch.from_numpy(idx)
    out = torch.zeros(int(torch.prod(torch.tensor(payload["shape"]))), device=device)
    out[pos] = q.float() * payload["scale"]
    return out.view(payload["shape"])


def payload_bytes(payload: dict) -> int:
    q = np.asarray(payload["q"])
    idx = np.asarray(payload["idx"])
    counts = np.asarray(payload["counts"])
    return int(q.nbytes + idx.nbytes + counts.nbytes)


def serialize_payload(payload: dict) -> bytes:
    import io
    import numpy as np
    buf = io.BytesIO()
    np.savez(
        buf,
        shape=np.array(payload["shape"], dtype=np.int64),
        scale=np.array(payload["scale"], dtype=np.float64),
        q=payload["q"].numpy(),
        idx=np.asarray(payload["idx"]),
        counts=payload["counts"].numpy(),
    )
    return buf.getvalue()


def unwrap_transport(raw) -> bytes:
    """Undo the Flower compat transport envelope (ndarray / .npy stream) -> raw bytes."""
    import io
    import numpy as np
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    if isinstance(raw, bytes) and raw[:6] == b"\x93NUMPY":
        raw = np.load(io.BytesIO(raw)).tobytes()
    return raw


def deserialize_payload(raw) -> dict:
    import io
    import numpy as np
    raw = unwrap_transport(raw)
    buf = io.BytesIO(raw)
    with np.load(buf) as z:
        return {
            "shape": tuple(int(s) for s in z["shape"]),
            "scale": float(z["scale"]),
            "q": torch.from_numpy(z["q"].copy()),
            "idx": torch.from_numpy(z["idx"].copy()),
            "counts": torch.from_numpy(z["counts"].copy()),
        }
