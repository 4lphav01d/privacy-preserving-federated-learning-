"""Opacus DP-SGD engine: client-side per-sample clipping, Gaussian noise, privacy ledger.

Memory bound: the logical batch of 64 is executed as physical micro-batches of 8
via Opacus BatchMemoryManager, so the per-sample gradient term never materialises
for the full logical batch. The privacy accountant runs over the logical steps.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CLIP_NORM = 1.0


def attach_dp(model: nn.Module, optimizer: torch.optim.Optimizer, train_loader,
              total_epochs: int, target_epsilon: float, delta: float):
    """Attach Opacus PrivacyEngine calibrated for the WHOLE client budget.

    total_epochs = rounds * local_epochs (the accountant sees every logical step).
    The engine persists across rounds so all noise draws consume one (eps, delta).
    """
    from opacus.privacy_engine import PrivacyEngine

    engine = PrivacyEngine()
    model, optimizer, train_loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=total_epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=CLIP_NORM,
    )
    logger.info("DP attached: sigma=%.4f for target eps=%.3f, delta=%.2e",
                optimizer.noise_multiplier, target_epsilon, delta)
    return engine, model, optimizer, train_loader


def achieved_epsilon(engine, delta: float) -> float:
    return engine.get_epsilon(delta=delta)


def dp_epoch(model, train_loader, optimizer, criterion, device, physical_batch: int = 8):
    """One local epoch: logical batches from the DP loader, physical micro-batches of 8."""
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    model.train()
    total_loss, seen = 0.0, 0
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=physical_batch,
        optimizer=optimizer,
    ) as memory_safe_loader:
        for x, y in memory_safe_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            seen += len(x)
    return total_loss / max(seen, 1)


def plain_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        seen += len(x)
    return total_loss / max(seen, 1)
