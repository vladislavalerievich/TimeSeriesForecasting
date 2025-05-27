from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.synthetic_generation.generator_params import GeneratorParams


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series generators.

    All concrete implementations must define `generate_time_series`.
    """

    @abstractmethod
    def generate_time_series(
        self, random_seed: Optional[int] = None, periodicity: str = "D"
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic time series data.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.
        periodicity : str, optional
            Time unit for timestamp generation. Examples: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'.

        Returns
        -------
        dict
            Dictionary containing:
            - 'timestamps': Array of np.datetime64 values
            - 'values': Array of shape (length, num_channels) or (length,)
        """
        pass


class GeneratorWrapper:
    """
    Unified base class for all generator wrappers, using a GeneratorParams dataclass
    for configuration. Provides parameter sampling, validation, and batch formatting utilities.
    """

    def __init__(self, params: GeneratorParams):
        """
        Initialize the GeneratorWrapper with a GeneratorParams dataclass.

        Parameters
        ----------
        params : GeneratorParams
            Dataclass instance containing all generator configuration parameters.
        """
        self.params = params
        self._set_random_seeds(self.params.global_seed)
        self._validate_input_parameters()

    def _set_random_seeds(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _validate_input_parameters(self) -> None:
        tuple_params = {
            "history_length": self.params.history_length,
            "target_length": self.params.target_length,
            "num_channels": self.params.num_channels,
        }
        for param_name, param_value in tuple_params.items():
            if isinstance(param_value, tuple):
                min_val, max_val = param_value
                if min_val > max_val:
                    raise ValueError(
                        f"For parameter '{param_name}', the minimum value ({min_val}) "
                        f"cannot exceed the maximum value ({max_val})"
                    )

    def _parse_param_value(
        self,
        param_config: Union[int, float, Tuple[int, float], Tuple[float, float]],
        is_int: bool = True,
    ) -> Union[int, float]:
        if isinstance(param_config, (int, float)):
            return int(param_config) if is_int else float(param_config)
        min_val, max_val = param_config
        return self._sample_from_range(min_val, max_val, is_int)

    def _sample_from_range(
        self,
        min_val: Union[int, float],
        max_val: Union[int, float],
        is_int: bool = True,
    ) -> Union[int, float]:
        if min_val == max_val:
            return min_val
        if self.params.distribution_type == "uniform":
            value = np.random.uniform(min_val, max_val)
        elif self.params.distribution_type == "log_uniform":
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.params.distribution_type}"
            )
        return int(value) if is_int else value

    def sample_parameters(self) -> Dict[str, Any]:
        history_length = self._parse_param_value(self.params.history_length)
        target_length = self._parse_param_value(self.params.target_length)
        num_channels = self._parse_param_value(self.params.num_channels)
        periodicity = np.random.choice(self.params.periodicities)
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
        history_values = torch.tensor(
            values[:, :history_length, :], dtype=torch.float32
        )
        future_values = torch.tensor(
            values[:, history_length : history_length + target_length, :],
            dtype=torch.float32,
        )
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
    ):
        from src.data_handling.data_containers import BatchTimeSeriesContainer
        from src.data_handling.time_features import compute_time_features

        history_values, future_values, history_timestamps, target_timestamps = (
            self._split_time_series_data(
                values, timestamps, history_length, target_length
            )
        )
        history_time_features = compute_time_features(
            history_timestamps, include_subday=True
        )
        target_time_features = compute_time_features(
            target_timestamps, include_subday=True
        )
        target_index = torch.randint(0, num_channels, (batch_size,))
        target_values = torch.zeros((batch_size, target_length), dtype=torch.float32)
        for i in range(batch_size):
            target_values[i] = future_values[i, :, target_index[i]]
        return BatchTimeSeriesContainer(
            history_values=history_values,
            target_values=target_values,
            target_index=target_index,
            history_time_features=history_time_features,
            target_time_features=target_time_features,
            static_features=None,
            history_mask=None,
            target_mask=None,
        )

    def generate_batch(self, batch_size: int, seed: Optional[int] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement generate_batch()")
