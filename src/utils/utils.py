import datetime
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
        f"num_epochs_{config['num_epochs']}_"
        f"num_training_iterations_per_epoch_{config['num_training_iterations_per_epoch']}_"
        f"num_encoder_layers_{config['TimeSeriesModel']['num_encoder_layers']}_"
        f"initial_lr{config['initial_lr']}_"
        f"learning_rate_{config['learning_rate']}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
