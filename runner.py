import torch as t
import itertools
import copy
from typing import List, Dict, Any
from configs import ExperimentConfig, GridSearchConfig
from experiment_builder import ExperimentBuilder
from comparative_trainer import ComparativeTrainer


def expand_grid(grid_config: GridSearchConfig) -> List[ExperimentConfig]:
    """Expands a GridSearchConfig into a list of ExperimentConfigs."""

    keys, values = zip(*grid_config.param_grid.items())
    experiments = []

    for combination in itertools.product(*values):
        new_config = copy.deepcopy(grid_config.base_config)

        # Create name
        param_str = "_".join(
            f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combination)
        )
        new_config.name = f"{grid_config.name}_{param_str}"

        # Set attributes dynamically
        for key, value in zip(keys, combination):
            target = new_config
            parts = key.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)

        experiments.append(new_config)

    return experiments


def run_sweep(grid_config: GridSearchConfig) -> List[Dict[str, Any]]:
    configs = expand_grid(grid_config)
    results = []
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    print(f"--- Running {len(configs)} experiments for sweep: {grid_config.name} ---")

    # If no data params change, generate data once.
    data_params_varied = any("data_config" in k for k in grid_config.param_grid.keys())
    shared_data = None

    if not data_params_varied:
        shared_data = ExperimentBuilder.get_data(grid_config.base_config)

    for i, cfg in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {cfg.name}")

        trainer = ExperimentBuilder.build_trainer(
            cfg, device, pre_generated_data=shared_data
        )
        trainer.train()

        results.append(
            {"config_name": cfg.name, "config": cfg, "history": trainer.history}
        )

    return results


def run_comparative(
    config_a: ExperimentConfig, config_b: ExperimentConfig, steps: int
) -> Dict[str, Any]:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"--- Comparative Run: {config_a.name} vs {config_b.name} ---")

    # For comparison, we force shared data based on config_a's settings
    shared_data = ExperimentBuilder.get_data(config_a)

    trainer_a = ExperimentBuilder.build_trainer(
        config_a, device, pre_generated_data=shared_data
    )
    trainer_b = ExperimentBuilder.build_trainer(
        config_b, device, pre_generated_data=shared_data
    )

    comp_trainer = ComparativeTrainer(trainer_a, trainer_b, num_steps=steps)
    history = comp_trainer.train()

    return {"config_a": config_a, "config_b": config_b, "history": history}
