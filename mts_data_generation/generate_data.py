import torch
from mlp import MLP
from utils import default_device
from distributions import parse_distributions, sample_distributions
from visualize import visualize_time_series


class MLPPriorTimeSeries:
    def __init__(self, config=None, time_steps=10, sampling="normal"):
        self.config = parse_distributions(config or {})
        self.time_steps = time_steps  # Number of time steps per sequence
        self.sampling = sampling  # Define sampling type (e.g., "normal", "uniform", "mixed")

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1):
        """
        Generate a batch of multivariate time series data.

        Args:
            batch_size (int): Number of sequences in a batch.
            n_samples (int): Number of samples per call.
            num_features (int): Number of features per time step.
            device (str): Torch device (CPU/GPU).
            num_outputs (int): Number of target variables.

        Returns:
            x (Tensor): Input time series of shape (batch_size, time_steps, num_features).
            y (Tensor): Target values of shape (batch_size, time_steps, num_outputs).
        """
        sampled_config = sample_distributions(self.config)

        # Ensure all required arguments are included
        mlp_kwargs = {
            "num_layers": sampled_config.get("num_layers", 3),
            "prior_mlp_hidden_dim": sampled_config.get("prior_mlp_hidden_dim", 32),
            "prior_mlp_activations": sampled_config.get("prior_mlp_activations", torch.nn.ReLU),
            "noise_std": sampled_config.get("noise_std", 0.1),
            "y_is_effect": sampled_config.get("y_is_effect", False),
            "pre_sample_weights": sampled_config.get("pre_sample_weights", False),
            "prior_mlp_dropout_prob": sampled_config.get("prior_mlp_dropout_prob", 0.1),
            "pre_sample_causes": sampled_config.get("pre_sample_causes", False),
            "prior_mlp_scale_weights_sqrt": sampled_config.get("prior_mlp_scale_weights_sqrt", False),
            "random_feature_rotation": sampled_config.get("random_feature_rotation", False),
            "add_uninformative_features": sampled_config.get("add_uninformative_features", False),
            "is_causal": sampled_config.get("is_causal", True),
            "num_causes": sampled_config.get("num_causes", num_features),
            "block_wise_dropout": sampled_config.get("block_wise_dropout", False),
            "init_std": sampled_config.get("init_std", 0.02),
            "sort_features": sampled_config.get("sort_features", False),
            "in_clique": sampled_config.get("in_clique", False)
        }

        sample = [
            [
                MLP(
                    device, num_features, num_outputs, n_samples, self.sampling, **mlp_kwargs
                ).to(device)()
                for _ in range(self.time_steps)
            ]
            for _ in range(batch_size)
        ]

        # Unpack time-series sequences
        x, y = zip(*[list(zip(*s)) for s in sample])

        # Convert to tensors
        x = torch.stack([torch.cat(seq, dim=1) for seq in x], dim=0)  # (batch_size, time_steps, num_features)
        y = torch.stack([torch.cat(seq, dim=1) for seq in y], dim=0)  # (batch_size, time_steps, num_outputs)

        return x, y


if __name__ == "__main__":
    time_series_generator = MLPPriorTimeSeries(config={"prior_mlp_hidden_dim": 32}, time_steps=50)

    # Generate synthetic time-series data
    x, y = time_series_generator.get_batch(batch_size=16, n_samples=1, num_features=5, num_outputs=2)

    print(x.shape)  # Expected: (16, 1, 20, 5) --> 16 sequences, 1 sample, 50 time steps, 5 features
    print(y.shape)  # Expected: (16, 1. 20, 2) --> 16 sequences, 1 sample, 50 time steps, 2 target values

    visualize_time_series(x, y, num_sequences=3)  # Visualize the generated time-series data
