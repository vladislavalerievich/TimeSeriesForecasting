from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series generators.

    All concrete implementations must define `generate_time_series` and `length`.
    """

    @property
    @abstractmethod
    def length(self) -> int:
        """
        Length of the generated time series.
        """
        pass

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


class AbstractGeneratorWrapper:
    """
    Abstract base class for generator wrapper implementations.
    Defines a common interface for all synthetic data generator wrappers.
    """

    def __init__(
        self,
        global_seed: int = 42,
        distribution_type: str = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
        num_channels: Union[int, Tuple[int, int]] = (1, 256),
        **kwargs,
    ):
        """
        Initialize the AbstractGeneratorWrapper.

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
        """
        self.global_seed = global_seed
        self.distribution_type = distribution_type
        self.history_length = history_length
        self.target_length = target_length
        self.num_channels = num_channels

        # Set random seeds
        self._set_random_seeds(self.global_seed)

    def _set_random_seeds(self, seed: int) -> None:
        """
        Set random seeds for numpy and torch for reproducibility.

        Parameters
        ----------
        seed : int
            The random seed to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _parse_param_value(
        self,
        param_config: Union[int, float, Tuple[int, float], Tuple[float, float]],
        is_int: bool = True,
    ) -> Union[int, float]:
        """
        Parse a parameter configuration which can be either a fixed value or a range.
        If it's a fixed value, return it directly; if it's a range, sample from it.

        Parameters
        ----------
        param_config : Union[int, float, Tuple[int, float], Tuple[float, float]]
            Parameter configuration, either a fixed value or a (min, max) range.
        is_int : bool, optional
            Whether to return an integer value (default: True).

        Returns
        -------
        Union[int, float]
            The fixed value or a sampled value from the range.
        """
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
        """
        Sample a value from the specified range using the configured distribution type.

        Parameters
        ----------
        min_val : Union[int, float]
            Minimum value of the range.
        max_val : Union[int, float]
            Maximum value of the range.
        is_int : bool, optional
            Whether to return an integer value (default: True).

        Returns
        -------
        Union[int, float]
            A sampled value from the specified range.
        """
        if min_val == max_val:
            return min_val

        if self.distribution_type == "uniform":
            value = np.random.uniform(min_val, max_val)
        elif self.distribution_type == "log_uniform":
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return int(value) if is_int else value

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for a batch generation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        raise NotImplementedError("Subclasses must implement sample_parameters()")

    def generate_batch(
        self, batch_size: int, seed: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a batch of synthetic multivariate time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed for this batch (default: None).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the generated batch data.
        """
        raise NotImplementedError("Subclasses must implement generate_batch()")

    def format_to_container(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
        batch_size: int,
        num_channels: int,
    ) -> Any:
        """
        Format generated time series data into a container format.

        Parameters
        ----------
        values : np.ndarray
            Generated time series values.
        timestamps : np.ndarray
            Generated timestamps.
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
        Any
            A container with the formatted time series data.
        """
        raise NotImplementedError("Subclasses must implement format_to_container()")
