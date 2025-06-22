import os
import random

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_descriptive_model_name(config):
    return (
        f"{config['model_name']}_"
        f"batch_size_{config['batch_size']}_"
        f"num_epochs_{config['num_epochs']}_"
        f"initial_lr{config['initial_lr']}_"
        f"learning_rate_{config['learning_rate']}_"
    )


def avoid_constant_inputs(inputs, outputs):
    idx_const_in = torch.nonzero(
        torch.all(inputs == inputs[:, 0].unsqueeze(1), dim=1)
    ).squeeze(1)
    if idx_const_in.size(0) > 0:
        inputs[idx_const_in, 0] += np.random.uniform(0.1, 1)
