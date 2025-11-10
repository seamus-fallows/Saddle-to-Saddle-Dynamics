#%%
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Literal, List, Dict, Union, Optional, Tuple, Sequence
from jaxtyping import Int, Float
import numpy as np
import torch as t
from IPython.display import display
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch import optim


device = t.device(
    "cuda" if t.cuda.is_available() else "cpu"
)


MAIN = __name__ == "__main__"

#%%
@dataclass
class DeepLinearNetworkConfig:
    num_hidden: int
    hidden_size: int
    in_size: int
    out_size: int
    weight_var: float = 1.0

@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    lr: float = 0.01
    seed: int = 0
    optimizer_cls: type[optim.Optimizer] = optim.SGD
    criterion_cls: type[nn.Module] = nn.MSELoss


class DeepLinearNetwork(nn.Module):
    def __init__(self, config: DeepLinearNetworkConfig):
        super().__init__()
        self.config = config
        in_size, out_size, hidden_size, num_hidden = (
            config.in_size,
            config.out_size,
            config.hidden_size,
            config.num_hidden,
        )

        # Build model (bias=False for all layers)
        sizes = [in_size] + [hidden_size] * num_hidden + [out_size]
        self.model = nn.Sequential(
            *[nn.Linear(sizes[i], sizes[i + 1], bias=False) for i in range(len(sizes) - 1)]
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        std = self.config.weight_var ** 0.5

        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)

        self.apply(init_fn)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)



class DeepLinearNetworkTrainer:
    def __init__(self, model: DeepLinearNetwork, config: TrainingConfig, train_set: Dataset, test_set: Dataset):
        self.model = model.to(device)
        self.config = config
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = config.optimizer_cls(self.model.parameters(), lr=config.lr)
        self.criterion = config.criterion_cls()
        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
        self.history = {
            "train_loss": [],
            "test_loss": [],
        }

    def evaluate(self) -> float:
        """Compute average loss on the test set."""
        self.model.eval()
        total_loss = 0.0
        n_examples = 0

        with t.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                n_examples += batch_size

        self.model.train()
        return total_loss / n_examples

    def training_step(self, x: Tensor, y: Tensor) -> Tensor:
        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss

    def train(self) -> DeepLinearNetwork:
        """Performs a full training run."""
        self.step = 0
        for epoch in range(self.config.num_epochs):
            epoch_train_loss = 0.0
            n_train = 0
            progress_bar = tqdm(self.train_loader, total=int(len(self.train_loader)), ascii=True)
            progress_bar.set_description(f"Epoch {epoch+1}/{self.config.num_epochs}")
            for x, y in progress_bar:
                x, y = x.to(device), y.to(device)
                loss = self.training_step(x, y)
                batch_size = x.size(0)
                epoch_train_loss += loss.item() * batch_size
                n_train += batch_size

                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = epoch_train_loss / n_train
            avg_test_loss = self.evaluate()

            self.history["train_loss"].append(avg_train_loss)
            self.history["test_loss"].append(avg_test_loss)
        return self.model


