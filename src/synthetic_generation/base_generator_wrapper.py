from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.time_features import compute_time_features
from src.synthetic_generation.constants import DEFAULT_START_DATE
from synthetic_generation.abstract_classes import AbstractGeneratorWrapper


class BaseGeneratorWrapper(AbstractGeneratorWrapper):
    """
    Base implementation of generator wrapper with common functionality.
    Provides functionality shared by all generator implementations.
    """

    def __init__(
        self,
        global_seed: int = 42,
        distribution_type: str = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
        num_channels: Union[int, Tuple[int, int]] = (1, 256),
        periodicities: List[str] = None,
        **kwargs,
    ):
        """
        Initialize the BaseGeneratorWrapper.

        Parameters
        ----------
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        distribution_type : str, optional
            Type of distribution to use for sampling parameters ("uniform" or "log_uniform", default: "uniform").
        history_length : Union[int, Tuple[int, int]], optional
            Fixed history length or range (min, max) (default: (64, 256)).
        target_length : Union[int, Tuple[int, int]], optional
            Fixed target length or range (min, max) (default: (32, 256)).
        num_channels : Union[int, Tuple[int, int]], optional
            Fixed number of channels or range (min, max) (default: (1, 256)).
        periodicities : List[str], optional
            List of possible periodicities to sample from (default: ["s", "m", "h", "D", "W"]).
        """
        super().__init__(
            global_seed=global_seed,
            distribution_type=distribution_type,
            history_length=history_length,
            target_length=target_length,
            num_channels=num_channels,
            **kwargs,
        )

        self.periodicities = (
            periodicities if periodicities is not None else ["s", "m", "h", "D", "W"]
        )
        self._validate_input_parameters()

    def _validate_input_parameters(self) -> None:
        """
        Validate input parameters to ensure they have valid values.
        """
        # Dictionary of parameters to validate
        tuple_params = {
            "history_length": self.history_length,
            "target_length": self.target_length,
            "num_channels": self.num_channels,
        }

        # Check each tuple parameter to ensure min < max
        for param_name, param_value in tuple_params.items():
            if isinstance(param_value, tuple):
                min_val, max_val = param_value
                if min_val > max_val:
                    raise ValueError(
                        f"For parameter '{param_name}', the minimum value ({min_val}) "
                        f"cannot exceed the maximum value ({max_val})"
                    )

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        history_length = self._parse_param_value(self.history_length)
        target_length = self._parse_param_value(self.target_length)
        num_channels = self._parse_param_value(self.num_channels)
        periodicity = np.random.choice(self.periodicities)

        return {
            "history_length": history_length,
            "target_length": target_length,
            "num_channels": num_channels,
            "periodicity": periodicity,
        }

    def _split_time_series_data(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Split time series data into history and target components.

        Parameters
        ----------
        values : np.ndarray
            Time series values with shape (batch_size, total_length, num_channels).
        timestamps : np.ndarray
            Timestamps with shape (batch_size, total_length).
        history_length : int
            Length of the history window.
        target_length : int
            Length of the target window.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
            Tuple containing (history_values, future_values, history_timestamps, target_timestamps).
        """
        # Split values into history and target
        history_values = torch.tensor(
            values[:, :history_length, :], dtype=torch.float32
        )
        future_values = torch.tensor(
            values[:, history_length : history_length + target_length, :],
            dtype=torch.float32,
        )

        # Split timestamps
        history_timestamps = timestamps[:, :history_length]
        target_timestamps = timestamps[
            :, history_length : history_length + target_length
        ]

        return history_values, future_values, history_timestamps, target_timestamps

    def format_to_container(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
        batch_size: int,
        num_channels: int,
    ) -> BatchTimeSeriesContainer:
        """
        Format generated time series data into a BatchTimeSeriesContainer.

        Parameters
        ----------
        values : np.ndarray
            Generated time series values with shape (batch_size, total_length, num_channels).
        timestamps : np.ndarray
            Generated timestamps with shape (batch_size, total_length).
        history_length : int
            Length of the history window.
        target_length : int
            Length of the target window.
        batch_size : int
            Number of time series in the batch.
        num_channels : int
            Number of channels in each time series.

        Returns
        -------
        BatchTimeSeriesContainer
            A container with the formatted time series data.
        """
        # Split data into history and target components
        history_values, future_values, history_timestamps, target_timestamps = (
            self._split_time_series_data(
                values, timestamps, history_length, target_length
            )
        )

        # Prepare time features
        history_time_features = compute_time_features(
            history_timestamps, include_subday=True
        )
        target_time_features = compute_time_features(
            target_timestamps, include_subday=True
        )

        # Randomly select target channel indices
        target_index = torch.randint(0, num_channels, (batch_size,))

        # Get the target values for each batch item using the selected target indices
        target_values = torch.zeros((batch_size, target_length), dtype=torch.float32)
        for i in range(batch_size):
            target_values[i] = future_values[i, :, target_index[i]]

        return BatchTimeSeriesContainer(
            history_values=history_values,
            target_values=target_values,
            target_index=target_index,
            history_time_features=history_time_features,
            target_time_features=target_time_features,
            static_features=None,  # Not used for now
            history_mask=None,  # Not used for now
            target_mask=None,  # Not used for now
        )

    def generate_batch(
        self, batch_size: int, seed: Optional[int] = None, **kwargs
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : Optional[int], optional
            Random seed for this batch (default: None).

        Returns
        -------
        BatchTimeSeriesContainer
            A container with the generated time series data.
        """
        raise NotImplementedError("Subclasses must implement generate_batch()")
