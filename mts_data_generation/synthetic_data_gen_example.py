import torch
import torch.nn as nn
import numpy as np
from mlp import MLPPrior

# Configuration for generating time series-like data
config = {
    'num_layers': {'distribution': 'meta_choice',
                   'choice_values': [2, 3, 4, 5]},
    'prior_mlp_hidden_dim': {'distribution': 'uniform_int', 'min': 32, 'max': 128},
    'noise_std': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
    'sampling': {'distribution': 'meta_choice',
                 'choice_values': ['normal', 'mixed', 'uniform']},
    'is_causal': {'distribution': 'meta_choice',
                  'choice_values': [True, False]},

    # Modifications to activation function selection
    'prior_mlp_activations': {'distribution': 'meta_choice',
                              'choice_values': [
                                  lambda: nn.ReLU(),
                                  lambda: nn.SiLU(),
                                  lambda: nn.Tanh()
                              ]},
    'y_is_effect': True,
    'pre_sample_weights': False,
    'prior_mlp_dropout_prob': 0.1,
    'pre_sample_causes': False,
    'prior_mlp_scale_weights_sqrt': True,
    'random_feature_rotation': True,
    'add_uninformative_features': True,
    'num_causes': None,  # Will be set to num_features if not specified
    'block_wise_dropout': False,
    'init_std': 0.1,
    'sort_features': False,
    'in_clique': False
}

# Create a prior for generating data
mlp_prior = MLPPrior(config)


# Generate synthetic time series data
def generate_synthetic_timeseries(
        num_features=10,  # Number of time series
        num_samples=1000,  # Length of each time series
        num_outputs=1,  # Optional target variable
        device='cpu'
):
    # Prepare config with explicit num_causes
    config_copy = config.copy()
    config_copy['num_causes'] = num_features

    # Create MLPPrior with modified config
    local_prior = MLPPrior(config_copy)

    # Generate batch
    x, y, _ = local_prior.get_batch(
        batch_size=1,  # Generate one batch
        n_samples=num_samples,
        num_features=num_features,
        num_outputs=num_outputs,
        device=device
    )

    return x.numpy(), y.numpy()


# Generate the synthetic data
X, y = generate_synthetic_timeseries()

print("Generated Synthetic Time Series:")
print("Shape of X (features):", X.shape)
print("Shape of y (optional target):", y.shape)
print("\nSample statistics:")
print("Mean of X:", X.mean(axis=0))
print("Std of X:", X.std(axis=0))