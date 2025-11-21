from dataclasses import dataclass
from typing import List, Any, Dict
from registries import register_dataset

# -----------------------------------------------------------------------------
# 1. Core Component Configs
# -----------------------------------------------------------------------------


@dataclass
class DeepLinearNetworkConfig:
    num_hidden: int
    hidden_size: int
    in_size: int
    out_size: int
    gamma: float | None  # Controls weight initialization std = hidden_size^(-gamma/2)
    bias: bool = False


@dataclass
class TrainingConfig:
    lr: float
    max_steps: int
    evaluate_every: int
    optimizer_name: str = "SGD"
    criterion_name: str = "MSELoss"
    batch_size: int | None = None  # None means full batch training


# -----------------------------------------------------------------------------
# 2. Data Configs
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    name: str
    num_samples: int
    data_seed: int
    in_size: int
    out_size: int
    test_split: float | None = None


@register_dataset("diagonal_teacher")
@dataclass
class DiagonalTeacherConfig(DataConfig):
    scale_factor: float = 1.0


@register_dataset("random_teacher")
@dataclass
class RandomTeacherConfig(DataConfig):
    scale_factor: float = 1.0
    mean: float = 0.0
    std: float = 1.0


# -----------------------------------------------------------------------------
# 3. Experiment Configs
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Defines a single experiment.
    """

    name: str
    # These override the in_size and out_size in data_config and DLN config to ensure consistency
    input_dim: int
    output_dim: int

    dln_config: DeepLinearNetworkConfig
    training_config: TrainingConfig
    data_config: DataConfig

    model_seed: int = 42


@dataclass
class GridSearchConfig:
    """
    Defines a sweep.
    param_grid keys should use dot notation (e.g. 'training_config.lr')
    """

    name: str
    base_config: ExperimentConfig
    param_grid: Dict[str, List[Any]]


@dataclass
class ComparativeExperimentConfig:
    """
    Defines a comparative experiment between two ExperimentConfigs.
    """

    name: str
    config_a: ExperimentConfig
    config_b: ExperimentConfig
    metric_names: List[str]
