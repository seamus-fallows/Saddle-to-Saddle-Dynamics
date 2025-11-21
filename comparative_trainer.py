from typing import List, Dict, Any, Optional
from DLN import DeepLinearNetworkTrainer
from registries import MetricFn


def train_comparative(
    trainer_a: DeepLinearNetworkTrainer,
    trainer_b: DeepLinearNetworkTrainer,
    num_steps: int,
    metrics: Optional[Dict[str, MetricFn]] = None,
    log_every: int = 1,
) -> List[Dict[str, Any]]:
    """
    Manages the training of two DeepLinearNetworkTrainer instances in lockstep.
    """
    history: List[Dict[str, Any]] = []

    for step in range(num_steps):
        # Step both trainers
        loss_a = trainer_a.training_step()
        loss_b = trainer_b.training_step()

        # Logging
        if step % log_every == 0 or step == num_steps - 1:
            test_loss_a = trainer_a.evaluate()
            test_loss_b = trainer_b.evaluate()

            log_entry = {
                "step": step + 1,
                "loss_a": loss_a,
                "loss_b": loss_b,
                "test_loss_a": test_loss_a,
                "test_loss_b": test_loss_b,
            }
            # Compute and log all provided metrics
            if metrics:
                for name, func in metrics.items():
                    log_entry[name] = func(trainer_a, trainer_b)

            history.append(log_entry)

    return history
