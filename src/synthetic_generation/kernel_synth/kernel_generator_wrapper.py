from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import KernelGeneratorParams
from src.synthetic_generation.kernel_synth.kernel_synth import KernelSynthGenerator


class KernelGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for KernelSynthGenerator to generate batches of multivariate time series data
    by stacking multiple univariate series. Accepts a KernelGeneratorParams dataclass for configuration.
    """

    def __init__(self, params: KernelGeneratorParams):
        super().__init__(params)
        self.params: KernelGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with KernelSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters()
        # Sample num_kernels
        if isinstance(self.params.num_kernels, tuple):
            num_kernels = self._sample_from_range(
                min_val=self.params.num_kernels[0],
                max_val=self.params.num_kernels[1],
                is_int=True,
            )
        else:
            num_kernels = self.params.num_kernels

        params.update({"max_kernels": num_kernels})
        return params

    def _generate_univariate_time_series(
        self,
        generator: KernelSynthGenerator,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a single univariate time series.

        Parameters
        ----------
        generator : KernelSynthGenerator
            Initialized generator instance.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Dict
            Dictionary containing the generated time series.
        """
        return generator.generate_time_series(random_seed=seed)

    def _generate_multivariate_time_series(
        self,
        num_channels: int,
        history_length: int,
        target_length: int,
        max_kernels: int,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a single multivariate time series by stacking multiple univariate series.

        Parameters
        ----------
        num_channels : int
            Number of channels (univariate series) to generate.
        history_length : int
            Length of the history part of each time series.
        target_length : int
            Length of the target part for the selected series.
        max_kernels : int
            Maximum number of kernels for generation.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]
            Tuple containing (history_values, target_values, start, target_index).
        """
        history_values = []
        target_values = None
        start = None
        target_index = np.random.randint(0, num_channels) if num_channels > 1 else 0
        total_length = history_length + target_length

        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            generator = KernelSynthGenerator(
                length=total_length if i == target_index else history_length,
                max_kernels=max_kernels,
            )
            result = self._generate_univariate_time_series(generator, channel_seed)
            if i == target_index:
                target_values = result["values"][-target_length:]
                history_values.append(result["values"][:history_length])
                if start is None:
                    start = result["start"]
            else:
                history_values.append(result["values"])

        history_values = (
            np.column_stack(history_values)
            if num_channels > 1
            else np.array(history_values[0])
        )
        return history_values, target_values, start, target_index

    def _generate_time_series_batch(
        self,
        batch_size: int,
        num_channels: int,
        length: int,
        max_kernels: int,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a batch of multivariate time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        num_channels : int
            Number of channels per time series.
        length : int
            Length of each time series.
        max_kernels : int
            Maximum number of kernels for generation.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (batch_values, batch_start).
        """
        batch_values = []
        batch_start = []
        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values, start = self._generate_multivariate_time_series(
                num_channels=num_channels,
                history_length=length,
                target_length=length,
                max_kernels=max_kernels,
                seed=batch_seed,
            )
            batch_values.append(values)
            batch_start.append(start)
        batch_values = np.array(batch_values).transpose(0, 1, 2)
        return batch_values, np.array(batch_start)

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using KernelSynthGenerator.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed for this batch (default: None).
        params : Dict[str, Any], optional
            Pre-sampled parameters to use. If None, parameters will be sampled.

        Returns
        -------
        BatchTimeSeriesContainer
            A container with the generated time series data.
        """
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters()
        history_length = params["history_length"]
        target_length = params["target_length"]
        num_channels = params["num_channels"]
        max_kernels = params["max_kernels"]
        total_length = history_length + target_length
        batch_history_values = []
        batch_target_values = []
        batch_start = []
        batch_target_index = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            if num_channels == 1:
                generator = KernelSynthGenerator(
                    length=total_length, max_kernels=max_kernels
                )
                result = self._generate_univariate_time_series(generator, batch_seed)
                history_values = result["values"][:history_length]
                target_values = result["values"][-target_length:]
                start = result["start"]
                target_index = 0
            else:
                history_values, target_values, start, target_index = (
                    self._generate_multivariate_time_series(
                        num_channels=num_channels,
                        history_length=history_length,
                        target_length=target_length,
                        max_kernels=max_kernels,
                        seed=batch_seed,
                    )
                )
            # Ensure shape for history_values: (seq_len, num_channels)
            if num_channels == 1:
                history_values = history_values.reshape(-1, 1)
            batch_history_values.append(history_values)
            batch_target_values.append(target_values)
            batch_start.append(start)
            batch_target_index.append(target_index)

        return self._format_to_container(
            history_values=batch_history_values,
            target_values=batch_target_values,
            target_index=np.array(batch_target_index),
            start=np.array(batch_start),
            frequency=None,
        )

    def _format_to_container(
        self,
        history_values: np.ndarray,
        target_values: np.ndarray,
        target_index: np.ndarray,
        start: np.ndarray,
        frequency: Optional[Frequency] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Format the generated time series data into a BatchTimeSeriesContainer.

        Parameters
        ----------
        history_values: np.ndarray
            Shape: [batch_size, history_length, num_channels]
        target_values: np.ndarray
            Shape: [batch_size, target_length] for both univariate and multivariate
        target_index: np.ndarray
            Shape: [batch_size], index of the target channel for multivariate data
        start: np.ndarray of np.datetime64
            Shape: [batch_size]
        frequency: Optional[Frequency]
            Frequency of the time series. If None, a random frequency is selected.
        """
        history_values_tensor = torch.tensor(history_values, dtype=torch.float32)
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32)
        target_index_tensor = torch.tensor(target_index, dtype=torch.long)

        if frequency is None:
            frequency = np.random.choice(list(Frequency))

        return BatchTimeSeriesContainer(
            history_values=history_values_tensor,
            target_values=target_values_tensor,
            target_index=target_index_tensor,
            start=start,
            frequency=frequency,
        )
