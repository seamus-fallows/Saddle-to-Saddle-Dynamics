#%%
import random
import torch as t
import numpy as np
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
from configs import DeepLinearNetworkConfig, TrainingConfig, TeacherStudentExperimentConfig
from student_teacher_data import generate_teacher_student_data

MAIN = __name__ == "__main__"
#%%

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

def run_once(exp: TeacherStudentExperimentConfig, run_id: int):
    model_config = exp.dln_config
    training_config = exp.training_config
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    seed = exp.base_seed + run_id
    _set_all_seeds(seed)

    # Generate data
    inputs, outputs = generate_teacher_student_data(
        num_samples=exp.num_samples,
        in_size=model_config.in_size,
        scale_factor=exp.teacher_matrix_scale_factor
    )
    
    # Split data into train and test sets
    n_train = int((1 - exp.test_split) * exp.num_samples)
    train_set = (inputs[:n_train], outputs[:n_train])
    test_set = (inputs[n_train:], outputs[n_train:])

    # Initialize model
    model = DeepLinearNetwork(model_config)
    model.init_weights(exp.gamma)
    trainer = DeepLinearNetworkTrainer(model, training_config, train_set, test_set, device)

    # Train model
    trainer.train()

    run_log = {
        "run_id": run_id,
        "seed": seed,
        "history": trainer.history,
    }
    return run_log

