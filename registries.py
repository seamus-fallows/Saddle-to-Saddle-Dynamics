import torch.nn as nn
import torch.optim as optim
from typing import Dict, Type, Callable, Any

MetricFn = Callable[[Any, Any], float]

# -----------------------------------------------------------------------------
# 1. Internal Storage
# -----------------------------------------------------------------------------
_CUSTOM_OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {}
_CUSTOM_CRITERIA: Dict[str, Type[nn.Module]] = {}
_CUSTOM_METRICS: Dict[str, MetricFn] = {}
_DATASET_REGISTRY: Dict[str, Type[Any]] = {}


# -----------------------------------------------------------------------------
# 2. Decorators
# -----------------------------------------------------------------------------
def register_optimizer(name: str):
    def decorator(cls):
        _CUSTOM_OPTIMIZERS[name] = cls
        return cls

    return decorator


def register_criterion(name: str):
    def decorator(cls):
        _CUSTOM_CRITERIA[name] = cls
        return cls

    return decorator


def register_metric(name: str):
    def decorator(func: MetricFn):
        _CUSTOM_METRICS[name] = func
        return func

    return decorator


def register_dataset(name: str):
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


# -----------------------------------------------------------------------------
# 3. Lookup Functions
# -----------------------------------------------------------------------------
def get_optimizer_cls(name: str) -> Type[optim.Optimizer]:
    # Check Custom Registry
    if name in _CUSTOM_OPTIMIZERS:
        return _CUSTOM_OPTIMIZERS[name]

    # Check Standard PyTorch
    if hasattr(optim, name):
        return getattr(optim, name)

    raise ValueError(f"Optimizer '{name}' not found in registry or torch.optim")


def get_criterion_cls(name: str) -> Type[nn.Module]:
    # Check Custom Registry
    if name in _CUSTOM_CRITERIA:
        return _CUSTOM_CRITERIA[name]

    # Check Standard PyTorch
    if hasattr(nn, name):
        return getattr(nn, name)

    raise ValueError(f"Criterion '{name}' not found in registry or torch.nn")


def get_metric_fn(name: str) -> MetricFn:
    if name in _CUSTOM_METRICS:
        return _CUSTOM_METRICS[name]
    raise ValueError(f"Metric '{name}' not found in registry.")


def get_dataset_config_cls(name: str) -> Any:
    if name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name]
    raise ValueError(f"Dataset config '{name}' not found in registry.")
