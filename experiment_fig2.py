#%%
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
import torch as t
from configs import DeepLinearNetworkConfig, TrainingConfig, TeacherStudentExperimentConfig
from runner import run_once

MAIN = __name__ == "__main__"
#%%
if MAIN:
    # Example experiment configuration
    dln_config = DeepLinearNetworkConfig(
        num_hidden=2,
        hidden_size=50,
        in_size=5,
        out_size=5,
        bias=False
    )

    training_config = TrainingConfig(
        num_epochs=200,
        lr=0.01,
        optimizer_cls=t.optim.SGD,
        criterion_cls=t.nn.MSELoss,
        evaluate_every=20,
        batch_size=None
    )

    experiment_config = TeacherStudentExperimentConfig(
        name="example_experiment",
        dln_config=dln_config,
        training_config=training_config,
        gamma=0.2,
        num_samples=100,
        teacher_matrix_scale_factor=10.0,
        test_split=0.2,
        base_seed=42
    )

    # Run a single experiment
    log = run_once(experiment_config, run_id=0)
    print(log)

