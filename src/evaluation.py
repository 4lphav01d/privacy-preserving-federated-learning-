"""Evaluation: global accuracy, and the gradient-inversion defence harness (DLG / Inverting Gradients)."""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def test_accuracy(model, loader, device="cpu") -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def _run_attack(model, x, y, grads_target, criterion, steps, lr, method: str) -> torch.Tensor:
    """Optimise a dummy input so its parameter-gradients match the intercepted update."""
    dummy = torch.distributions.Normal(0, 0.1).sample(x.shape).detach().requires_grad_(True)
    dummy_label = y.clone()  # label assistance off: use zeros for a harder attack
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam([dummy], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        preds = model(dummy)
        loss = criterion(preds, dummy_label.to(preds.device))
        grads = torch.autograd.grad(loss, params, create_graph=True)
        if method == "dlg":
            loss_attack = sum((g - gt).pow(2).sum() for g, gt in zip(grads, grads_target))
        else:  # "ig": cosine similarity on cossim-summed grads (Geiping et al.)
            loss_attack = -sum((g * gt).sum() / ((g.norm() + 1e-8) * (gt.norm() + 1e-8)) for g, gt in zip(grads, grads_target))
        loss_attack.backward()
        opt.step()
        scheduler.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    return dummy.detach().clamp(0, 1)


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).sum().sqrt())


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity
    a = a.squeeze().cpu().numpy()
    b = b.squeeze().cpu().numpy()
    if a.ndim == 3:
        vals = [structural_similarity(a[c], b[c], data_range=1.0) for c in range(a.shape[0])]
        return float(np.mean(vals))
    return float(structural_similarity(a, b, data_range=1.0))


def inversion_defence_experiment(cfg: dict, out_rows: list[dict]) -> list[dict]:
    """Attack the SAME client batch twice: once from a non-private one-step gradient,
    once from a DP-clipped+noised one-step gradient. Reports L2/SSIM for DLG and IG."""
    from .data import load_datasets, test_loader
    from .models import build_model
    from .dp_engine import CLIP_NORM

    set_dev = "cpu"
    model = build_model(cfg["dataset"]).to(set_dev)
    criterion = nn.CrossEntropyLoss()

    if cfg.get("capture"):  # real mid-training capture: trained params + the batch behind the update
        blob = torch.load(cfg["capture"], weights_only=False)
        from .utils import unflatten_to_state_dict
        unflatten_to_state_dict(blob["params"], model.state_dict())
        x_true, y_true = blob["x"], blob["y"]
    else:
        train_set, _ = load_datasets(cfg["dataset"])
        targets = cfg.get("attack_targets", [0, 1])
        per_class = cfg.get("attack_images_per_class", 2)
        xs, ys = [], []
        for t in targets:
            idx = np.where(np.array(train_set.targets) == t)[0][:per_class]
            for i in idx:
                x, y = train_set[i]
                xs.append(x)
                ys.append(torch.tensor(y))
        x_true = torch.stack(xs).to(set_dev)
        y_true = torch.stack(ys).to(set_dev)

    steps = cfg.get("attack_steps", 300)
    lr = cfg.get("attack_lr", 0.1)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)

    # non-private one-step gradients
    model.zero_grad()
    loss = criterion(model(x_true), y_true)
    grads_clean = [g.detach().clone() for g in torch.autograd.grad(loss, model.parameters())]

    # DP one-step gradients, computed exactly as DP-SGD does: per-sample clip to C,
    # sum over the batch, add Gaussian noise sigma * C * N(0, I), scale by 1/n.
    sigma = cfg.get("attack_sigma", 1.0)
    n = x_true.size(0)
    params = [p for p in model.parameters() if p.requires_grad]
    summed = [torch.zeros_like(p) for p in params]
    for i in range(n):
        model.zero_grad()
        loss_i = criterion(model(x_true[i:i + 1]), y_true[i:i + 1])
        gi = torch.autograd.grad(loss_i, params)
        norm = torch.sqrt(sum((g ** 2).sum() for g in gi))
        scale = min(1.0, float(CLIP_NORM) / (float(norm) + 1e-12))
        for acc, g in zip(summed, gi):
            acc += g.detach() * scale
    grads_dp = [(acc + sigma * float(CLIP_NORM) * torch.randn_like(acc)) / n for acc in summed]
    model.zero_grad()

    rows = []
    for method in ("dlg", "ig"):
        torch.manual_seed(seed)
        rec_clean = _run_attack(model, x_true, y_true, grads_clean, criterion, steps, lr, method)
        torch.manual_seed(seed)
        rec_dp = _run_attack(model, x_true, y_true, grads_dp, criterion, steps, lr, method)
        for cond, rec in (("non_private", rec_clean), ("dp", rec_dp)):
            rows.append({
                "dataset": cfg["dataset"], "method": method, "condition": cond,
                "sigma": sigma, "steps": steps,
                "l2": round(_l2(rec, x_true), 4),
                "ssim": round(_ssim(rec, x_true), 4),
            })
            logger.info("inversion %s %s: l2=%.4f ssim=%.4f", method, cond, rows[-1]["l2"], rows[-1]["ssim"])
    out_rows.extend(rows)
    return rows
