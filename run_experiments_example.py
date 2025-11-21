# %% [markdown]
# # Deep Linear Network Experiments
# This notebook demonstrates how to run experiments programmatically.

# %%
import os
import matplotlib.pyplot as plt
from config_loader import (
    load_experiment_config,
    load_grid_search_config,
    load_comparative_config,
)
from runner import run_single, run_sweep, run_comparative_experiment

# 1. Initialize the Extension System
# This registers your custom optimizers, criteria, and metrics.
import extensions

extensions.register_all()

# Helper to make paths work regardless of where you run this
BASE_DIR = "experiments"

# %% [markdown]
# ## 1. Single Run
# Loading `base.yaml` and training a single model.

# %%
config_path = os.path.join(BASE_DIR, "base.yaml")
print(f"Loading: {config_path}")

# Load
config = load_experiment_config(config_path)

# Run
result = run_single(config)

# Plot
history = result["history"]
train_loss = [x["train_loss"] for x in history]
test_loss = [x["test_loss"] for x in history]

plt.figure(figsize=(8, 4))
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss", linestyle="--")
plt.yscale("log")
plt.title(f"Single Run: {config.name}")
plt.legend()
plt.show()

# %% [markdown]
# ## 2. Parameter Sweep
# Loading `sweep.yaml` which inherits from `base.yaml`.

# %%
sweep_path = os.path.join(BASE_DIR, "sweep.yaml")
print(f"Loading Sweep: {sweep_path}")

# Load
sweep_config = load_grid_search_config(sweep_path)

# Run
sweep_results = run_sweep(sweep_config)

# Plot All Runs
plt.figure(figsize=(10, 6))
for res in sweep_results:
    loss = [h["train_loss"] for h in res["history"]]
    # The config name is automatically generated (e.g. sweep_lr=0.01...)
    plt.plot(loss, label=res["config_name"])

plt.yscale("log")
plt.title("Sweep Results")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Comparative Experiment
# Comparing SGD vs. SgdWithNoise using `compare.yaml`.

# %%
comp_path = os.path.join(BASE_DIR, "compare.yaml")
print(f"Loading Comparison: {comp_path}")

# Load
comp_config = load_comparative_config(comp_path)

# Run
comp_result = run_comparative_experiment(comp_config)
history = comp_result["history"]

# Plot Distance Metric
# We requested 'param_euclidean_dist' in the YAML
steps = [h["step"] for h in history]
dists = [h["param_euclidean_dist"] for h in history]

plt.figure(figsize=(8, 4))
plt.plot(steps, dists)
plt.title("Distance Between Models (SGD vs SgdWithNoise)")
plt.xlabel("Step")
plt.ylabel("Euclidean Distance")
plt.show()
