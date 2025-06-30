from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.synthetic_generation.common.utils import (
    select_safe_random_frequency,
    select_safe_start_date,
)
from src.synthetic_generation.generator_params import GeneratorParams


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series generators.

    All concrete implementations must define `generate_time_series`.
    """

    @abstractmethod
    def generate_time_series(
        self, random_seed: Optional[int] = None, periodicity: Frequency = Frequency.D
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic time series data.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.
        periodicity : Frequency, optional
            Time unit for timestamp generation. Defaults to Frequency.D (Daily).

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
        # For parameter sampling, we want diversity across batches even with similar seeds
        # Use a hash of the generator class name to ensure different generators get different parameter sequences
        param_seed = seed + hash(self.__class__.__name__) % 2**31
        self.rng = np.random.default_rng(param_seed)

        # Set global numpy and torch seeds for deterministic behavior in underlying generators
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _validate_input_parameters(self) -> None:
        tuple_params = {
            "future_length": self.params.future_length,
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

    def _parse_param_value(self, param: Union[int, Tuple[int, int], List[int]]) -> int:
        if isinstance(param, int):
            return param
        if isinstance(param, list):
            return self.rng.choice(param)
        if isinstance(param, tuple):
            min_val, max_val = param
            if min_val > max_val:
                raise ValueError(
                    f"Min value {min_val} cannot be greater than max value {max_val}"
                )
            if min_val == max_val:
                return min_val
            return self.rng.integers(low=min_val, high=max_val, size=1)[0]
        raise ValueError(f"Unsupported param type: {type(param)}")

    def _sample_from_range(
        self,
        min_val: Union[int, float],
        max_val: Union[int, float],
        is_int: bool = True,
    ) -> Union[int, float]:
        if min_val == max_val:
            return min_val
        if self.params.distribution_type == "uniform":
            value = self.rng.uniform(min_val, max_val)
        elif self.params.distribution_type == "log_uniform":
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            value = 10 ** self.rng.uniform(log_min, log_max)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.params.distribution_type}"
            )
        return int(value) if is_int else value

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameters with total_length fixed and history_length calculated.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values where
            history_length = total_length - future_length.
        """
        # Sample future_length
        future_length = self._parse_param_value(self.params.future_length)

        # Calculate history_length from total_length - future_length
        total_length = self.params.total_length
        history_length = total_length - future_length

        # Ensure history_length is positive
        if history_length <= 0:
            raise ValueError(
                f"history_length ({history_length}) must be positive. "
                f"total_length ({total_length}) - future_length ({future_length}) = {history_length}"
            )

        num_channels = self._parse_param_value(self.params.num_channels)

        # Select a suitable frequency based on the total length
        frequency = select_safe_random_frequency(total_length, self.rng)

        # Select a safe start date that prevents timestamp overflow
        if self.params.start is not None:
            start = self.params.start
        else:
            start = select_safe_start_date(
                history_length, future_length, frequency, self.rng
            )

        return {
            "total_length": total_length,
            "history_length": history_length,
            "future_length": future_length,
            "num_channels": num_channels,
            "frequency": frequency,
            "start": start,
        }

    def _format_to_container(
        self,
        values: np.ndarray,
        start: np.ndarray,
        history_length: int,
        future_length: int,
        frequency: Frequency,
    ) -> BatchTimeSeriesContainer:
        """
        Format the generated time series data into a BatchTimeSeriesContainer.

        Parameters
        ----------
        values: np.ndarray
            Shape: [batch_size, seq_len, num_channels]
        start: np.ndarray of np.datetime64
            Shape: [batch_size]
        history_length: int
            Length of the history window
        future_length: int
            Length of the future window
        frequency: Frequency
            Frequency of the time series.
        """

        # Split values into history and future
        history_values = torch.tensor(
            values[:, :history_length, :], dtype=torch.float32
        )
        future_values = torch.tensor(
            values[:, history_length : history_length + future_length, :],
            dtype=torch.float32,
        )

        return BatchTimeSeriesContainer(
            history_values=history_values,
            future_values=future_values,
            start=start,
            frequency=frequency,
        )

    def generate_batch(self, batch_size: int, seed: Optional[int] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement generate_batch()")
