import torch as t
from typing import Optional, Tuple
from torch import Tensor

from configs import ExperimentConfig
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
from data_utils import (
    generate_teacher_student_data,
    train_test_split,
    get_data_loaders,
    set_all_seeds,
)
from utils import initialize_dln_weights


class ExperimentBuilder:
    @staticmethod
    def get_data(
        config: ExperimentConfig,
    ) -> Tuple[Tuple[Tensor, Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Sets the data_seed, generates data, and returns train/test split.
        Does NOT affect model initialization random state.
        """
        # 1. Set Seed specifically for Data Generation
        set_all_seeds(config.data_config.seed)

        inputs, outputs = generate_teacher_student_data(
            num_samples=config.data_config.num_samples,
            in_size=config.data_config.in_size,
            scale_factor=config.data_config.teacher_matrix_scale_factor,
        )

        return train_test_split(inputs, outputs, config.data_config.test_split)

    @staticmethod
    def build_trainer(
        config: ExperimentConfig,
        device: t.device,
        pre_generated_data: Optional[Tuple] = None,
    ) -> DeepLinearNetworkTrainer:
        """
        Builds a trainer.
        - If pre_generated_data is None, generates new data using data_config.seed.
        - Sets model_seed before initializing weights.
        - Calculates weight_std based on gamma.
        """

        if pre_generated_data:
            train_set, test_set = pre_generated_data
        else:
            train_set, test_set = ExperimentBuilder.get_data(config)

        # Set Seed specifically for Model Init
        set_all_seeds(config.model_seed)

        # Get Data Loaders
        train_loader, test_loader = get_data_loaders(
            train_set, test_set, config.training_config.batch_size
        )

        # Initialize Model
        model = DeepLinearNetwork(config.dln_config)
        weight_std = config.dln_config.hidden_size ** (-config.gamma / 2)
        initialize_dln_weights(model, weight_std)

        return DeepLinearNetworkTrainer(
            model=model,
            config=config.training_config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
