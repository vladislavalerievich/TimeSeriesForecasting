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


def effective_rank(matrix):
    """
    Computes the effective rank of a matrix based on the definition by Roy and Vetterli.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        float: The effective rank of the matrix.
    """
    # Step 1: Compute singular values
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    # Handle cases with zero singular values for numerical stability
    singular_values = singular_values[
        singular_values > np.finfo(float).eps
    ]  # Use a small epsilon

    if len(singular_values) == 0:
        return 0  # Or handle as per convention for a zero matrix

    # Step 2: Calculate the singular value distribution (p_k)
    norm_s1 = np.sum(singular_values)
    if norm_s1 == 0:
        # This case implies all singular values were zero or very close to zero
        return 0  # Effective rank of a zero matrix can be considered 0
        # or handle as per paper's convention for all-zero matrix
        # The paper states "non-all-zero matrix A" [cite: 1]

    p_k = singular_values / norm_s1

    # Step 3: Calculate Shannon Entropy (H)
    # Handle p_k = 0 using the convention 0 * log(0) = 0
    # np.log(0) is -inf. p_k * np.log(p_k) would be 0 * -inf which is nan.
    # We can filter out p_k == 0 before calculating entropy terms
    # However, p_k calculated from non-zero singular_values should not be zero.
    # For robustness against extremely small p_k values that might cause issues:
    entropy_terms = -p_k * np.log(p_k)
    entropy = np.sum(entropy_terms)

    # Step 4: Compute Effective Rank
    erank = np.exp(entropy)

    return erank


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
