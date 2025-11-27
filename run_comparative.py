import json
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from dln.utils import seed_rng, get_device, to_device
from dln.data import create_dataset
from dln.comparative import ComparativeTrainer
from dln.factory import create_trainer


def run_comparative_experiment(cfg: DictConfig, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = Path(HydraConfig.get().runtime.output_dir)

    device = get_device()

    # Shared dataset
    seed_rng(cfg.data.data_seed)
    train_set, test_set = create_dataset(
        cfg.data,
        in_dim=cfg.model_a.in_dim,
        out_dim=cfg.model_a.out_dim,
    )

    # Move data
    train_inputs, train_targets = to_device(train_set, device)
    test_data = to_device(test_set, device) if test_set else None

    # Initialize Trainers

    # Model A
    trainer_a = create_trainer(
        model_cfg=cfg.model_a,
        training_cfg=cfg.training_a,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_data=test_data,
        device=device,
    )

    # Model B
    trainer_b = create_trainer(
        model_cfg=cfg.model_b,
        training_cfg=cfg.training_b,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_data=test_data,
        device=device,
    )

    comparative_trainer = ComparativeTrainer(
        trainer_a,
        trainer_b,
        max_steps=cfg.max_steps,
        evaluate_every=cfg.evaluate_every,
    )

    history = comparative_trainer.train(
        model_metrics=cfg.model_metrics,
        comparative_metrics=cfg.comparative_metrics,
    )

    # Save history
    history_path = output_dir / "history.jsonl"
    with history_path.open("w") as f:
        for record in history:
            json.dump(record, f)
            f.write("\n")

    return output_dir


@hydra.main(
    version_base=None,
    config_path="configs/comparative",
    config_name="diagonal_teacher",
)
def main(cfg: DictConfig) -> None:
    run_comparative_experiment(cfg)


if __name__ == "__main__":
    main()
