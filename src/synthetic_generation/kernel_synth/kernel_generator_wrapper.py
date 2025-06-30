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
        num_kernels = self._parse_param_value(self.params.num_kernels)
        params.update(
            {"max_kernels": num_kernels, "use_gpytorch": self.params.use_gpytorch}
        )
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
        length: int,
        max_kernels: int,
        use_gpytorch: bool,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a single multivariate time series by stacking multiple univariate series.

        Parameters
        ----------
        num_channels : int
            Number of channels (univariate series) to generate.
        length : int
            Length of the time series.
        max_kernels : int
            Maximum number of kernels for generation.
        use_gpytorch : bool
            Whether to use GPyTorch for generation.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        np.ndarray
            Shape: [seq_len, num_channels]
        """
        values = []
        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            generator = KernelSynthGenerator(
                length=length,
                max_kernels=max_kernels,
                use_gpytorch=use_gpytorch,
            )
            result = self._generate_univariate_time_series(generator, channel_seed)

            values.append(result)

        values = np.column_stack(values) if num_channels > 1 else np.array(values[0])
        return values

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

        num_channels = params["num_channels"]

        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=params["total_length"],
                max_kernels=params["max_kernels"],
                use_gpytorch=params["use_gpytorch"],
                seed=batch_seed,
            )
            # Ensure shape for values: (seq_len, num_channels)
            if num_channels == 1:
                values = values.reshape(-1, 1)
            batch_values.append(values)

        batch_values = np.array(batch_values)

        return self._format_to_container(
            values=batch_values,
            start=params["start"],
            history_length=params["history_length"],
            future_length=params["future_length"],
            frequency=params["frequency"],
        )

    def _format_to_container(
        self,
        values: np.ndarray,
        start: np.ndarray,
        history_length: int,
        future_length: int,
        frequency: Optional[Frequency] = None,
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
        future_length: int
        frequency: Optional[Frequency]
            Frequency of the time series. If None, a random frequency is selected.
        """
        return BatchTimeSeriesContainer(
            history_values=torch.tensor(
                values[:, :history_length, :], dtype=torch.float32
            ),
            future_values=torch.tensor(
                values[:, history_length : history_length + future_length, :],
                dtype=torch.float32,
            ),
            start=start,
            frequency=frequency,
        )
