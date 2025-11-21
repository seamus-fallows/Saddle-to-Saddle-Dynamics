import yaml
import os
from typing import Dict, Any
from configs import (
    ComparativeExperimentConfig,
    ExperimentConfig,
    DeepLinearNetworkConfig,
    TrainingConfig,
    GridSearchConfig,
)
from registries import get_dataset_config_cls


def load_experiment_config(path: str) -> ExperimentConfig:
    """
    Loads a YAML file, resolves inheritance,
    injects dimensions, and returns an ExperimentConfig object.
    """
    config_dict = _load_raw_yaml(path)

    # Inject top-level dimensions into sub-configs
    _inject_dimensions(config_dict)

    # Convert dictionary to Dataclasses
    return _build_experiment_config(config_dict)


def load_grid_search_config(path: str) -> GridSearchConfig:
    """
    Loads a GridSearch definition.
    """
    # Load the full YAML (resolves inheritance)
    data = _load_raw_yaml(path)

    # Extract Sweep-Specific Fields
    sweep_name = data.pop("name")
    param_grid = data.pop("param_grid")
    experiment_data = data

    # Ensure the base experiment has a name (reuse sweep name if needed)
    if "name" not in experiment_data:
        experiment_data["name"] = f"{sweep_name}_base"

    _inject_dimensions(experiment_data)

    return GridSearchConfig(
        name=sweep_name,
        base_config=_build_experiment_config(experiment_data),
        param_grid=param_grid,
    )


def load_comparative_config(path: str) -> ComparativeExperimentConfig:
    data = _load_raw_yaml(path)

    if "config_a" not in data or "config_b" not in data:
        raise ValueError(
            "Comparative config must have 'config_a' and 'config_b' sections."
        )

    # Process Config A
    dict_a = data["config_a"]
    _inject_dimensions(dict_a)
    config_a = _build_experiment_config(dict_a)

    # Process Config B
    dict_b = data["config_b"]
    _inject_dimensions(dict_b)
    config_b = _build_experiment_config(dict_b)

    return ComparativeExperimentConfig(
        name=data["name"],
        config_a=config_a,
        config_b=config_b,
        metric_names=data.get("metric_names", []),
    )


def _load_raw_yaml(path: str) -> Dict[str, Any]:
    """
    Loads YAML. If 'defaults_path' is present, loads the base recursively
    and merges the current config on top.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Handle Inheritance
    if "defaults_path" in config:
        base_path = config.pop("defaults_path")

        # Resolve relative path for the base config
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)

        base_config = _load_raw_yaml(base_path)

        # Merge: Current config overwrites base config
        _deep_merge(base_config, config)
        return base_config

    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursive dict merge. Modifies 'base' in place."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _inject_dimensions(config: Dict[str, Any]) -> None:
    """
    Reads input_dim/output_dim and pushes them into dln_config and data_config.
    """
    in_dim = config.get("input_dim")
    out_dim = config.get("output_dim")

    # Inject into DLN and Data Configs
    if "dln_config" in config:
        config["dln_config"]["in_size"] = in_dim
        config["dln_config"]["out_size"] = out_dim

    if "data_config" in config:
        config["data_config"]["in_size"] = in_dim
        config["data_config"]["out_size"] = out_dim


def _build_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
    """
    Constructs the Dataclass hierarchy from the dict.
    """
    required_keys = [
        "dln_config",
        "training_config",
        "data_config",
        "input_dim",
        "output_dim",
        "name",
    ]
    for k in required_keys:
        if k not in data:
            raise KeyError(
                f"Missing required config key: '{k}'. check your YAML structure or defaults."
            )

    training = TrainingConfig(**data["training_config"])
    dln = DeepLinearNetworkConfig(**data["dln_config"])

    # Build Data Config
    data_cfg_dict = data["data_config"]
    name = data_cfg_dict.get("name")

    data_cls = get_dataset_config_cls(name)
    data_obj = data_cls(**data_cfg_dict)

    exp_args = {
        "name": data["name"],
        "input_dim": data["input_dim"],
        "output_dim": data["output_dim"],
        "dln_config": dln,
        "training_config": training,
        "data_config": data_obj,
    }

    if "model_seed" in data:
        exp_args["model_seed"] = data["model_seed"]

    return ExperimentConfig(**exp_args)
