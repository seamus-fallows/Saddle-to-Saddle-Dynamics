from typing import List, Dict, Any, Callable
import torch as t
from .train import Trainer, run_training_loop

MetricFn = Callable[[Trainer, Trainer], float]


def param_distance(trainer_a: Trainer, trainer_b: Trainer) -> float:
    """Euclidean distance between flattened parameters of two models."""
    params_a = t.cat([p.view(-1) for p in trainer_a.model.parameters()])
    params_b = t.cat([p.view(-1) for p in trainer_b.model.parameters()])
    return t.norm(params_a - params_b, p=2).item()


METRICS: Dict[str, MetricFn] = {
    "param_distance": param_distance,
}


class ComparativeTrainer:
    """
    Trains two models in lockstep using the shared training loop.
    """

    def __init__(
        self,
        trainer_a: Trainer,
        trainer_b: Trainer,
        max_steps: int,
        metrics: List[str],
        evaluate_every: int = 1,
    ):
        self.trainer_a = trainer_a
        self.trainer_b = trainer_b
        self.max_steps = max_steps
        self.evaluate_every = evaluate_every
        self.metrics = metrics
        self.history: List[Dict[str, Any]] = []

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            name: METRICS[name](self.trainer_a, self.trainer_b) for name in self.metrics
        }

    def train(self) -> List[Dict[str, Any]]:
        self.trainer_a.model.train()
        self.trainer_b.model.train()

        def step_fn() -> Dict[str, float]:
            loss_a = self.trainer_a.training_step()
            loss_b = self.trainer_b.training_step()

            return {
                "train_loss_a": loss_a,
                "train_loss_b": loss_b,
            }

        def eval_fn() -> Dict[str, Any]:
            test_loss_a = self.trainer_a.evaluate()
            test_loss_b = self.trainer_b.evaluate()

            comp_metrics = self._compute_metrics()

            return {
                "test_loss_a": test_loss_a,
                "test_loss_b": test_loss_b,
                **comp_metrics,
            }

        # Train
        self.history = run_training_loop(
            max_steps=self.max_steps,
            evaluate_every=self.evaluate_every,
            step_fn=step_fn,
            eval_fn=eval_fn,
        )

        return self.history
