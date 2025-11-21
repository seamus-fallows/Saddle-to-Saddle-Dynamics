import torch as t
from typing import Optional, Tuple
from registries import get_optimizer_cls, get_criterion_cls

from configs import ExperimentConfig
from DLN import DeepLinearNetwork, DeepLinearNetworkTrainer
from data_utils import (
    create_dataset_from_config,
    get_data_loaders,
    set_all_seeds,
)


def build_trainer(
    config: ExperimentConfig,
    device: t.device,
    pre_generated_data: Optional[Tuple] = None,
) -> DeepLinearNetworkTrainer:
    if pre_generated_data:
        train_set, test_set = pre_generated_data
    else:
        train_set, test_set = create_dataset_from_config(config.data_config)

    set_all_seeds(config.model_seed)

    train_loader, test_loader = get_data_loaders(
        train_set, test_set, config.training_config.batch_size
    )

    model = DeepLinearNetwork(config.dln_config)

    opt_cls = get_optimizer_cls(config.training_config.optimizer_name)
    crit_cls = get_criterion_cls(config.training_config.criterion_name)

    return DeepLinearNetworkTrainer(
        model=model,
        config=config.training_config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer_cls=opt_cls,
        criterion_cls=crit_cls,
    )
