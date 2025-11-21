import argparse
import sys
import os
from config_loader import (
    load_experiment_config,
    load_grid_search_config,
    load_comparative_config,
    _load_raw_yaml,
)
from runner import run_single, run_sweep, run_comparative_experiment

import extensions


def main():
    parser = argparse.ArgumentParser(
        description="Deep Linear Network Experiment Runner"
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (cpu/cuda)."
    )
    args = parser.parse_args()
    extensions.register_all()

    path = args.config
    if not os.path.exists(path):
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)

    try:
        raw_data = _load_raw_yaml(path)
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)

    if "param_grid" in raw_data:
        config = load_grid_search_config(path)
        run_sweep(config)

    elif "config_a" in raw_data and "config_b" in raw_data:
        config = load_comparative_config(path)
        run_comparative_experiment(config)

    else:
        config = load_experiment_config(path)
        run_single(config)


if __name__ == "__main__":
    main()
