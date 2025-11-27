"""
Metrics for tracking during training.

All model metrics take an nn.Module and return a float or dict of floats.
Comparative metrics take two nn.Modules.
"""

import torch as t
from torch.nn import Module
from typing import Callable

# ============================================
# Single-Model Metrics
# ============================================


def weight_norm(model: Module) -> float:
    return t.cat([p.view(-1) for p in model.parameters()]).norm().item()


def gradient_norm(model: Module) -> float:
    """L2 norm of all gradients (call after backward, before optimizer step)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total**0.5


MODEL_METRICS: dict[str, Callable] = {
    "weight_norm": weight_norm,
    "gradient_norm": gradient_norm,
}

# ============================================
# Comparative Metrics
# ============================================


def param_distance(model_a: Module, model_b: Module) -> float:
    """Euclidean distance between flattened parameters of two models."""
    params_a = t.cat([p.view(-1) for p in model_a.parameters()])
    params_b = t.cat([p.view(-1) for p in model_b.parameters()])
    return t.norm(params_a - params_b, p=2).item()


COMPARATIVE_METRICS: dict[str, callable] = {
    "param_distance": param_distance,
}

# ============================================
# Helper Functions
# ============================================


def compute_model_metrics(model: Module, names: list[str]) -> dict[str, float]:
    """Compute a list of model metrics by name."""
    return {name: MODEL_METRICS[name](model) for name in names}


def compute_comparative_metrics(
    model_a: Module,
    model_b: Module,
    names: list[str],
) -> dict[str, float]:
    """Compute a list of comparative metrics by name."""
    return {name: COMPARATIVE_METRICS[name](model_a, model_b) for name in names}
