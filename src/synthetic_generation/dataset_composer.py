import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from src.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    LMCGeneratorParams,
    SineWaveGeneratorParams,
)
from src.synthetic_generation.gp_prior.gp_generator_wrapper import (
    GPGeneratorWrapper,
)
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)
from src.synthetic_generation.lmc_synth.lmc_generator_wrapper import (
    LMCGeneratorWrapper,
)
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetComposer:
    """
    Composes datasets from multiple generator wrappers according to specified proportions.
    Manages the generation of both training and validation datasets.
    """

    def __init__(
        self,
        generator_proportions: Optional[Dict[str, float]] = None,
        global_seed: int = 42,
    ):
        """
        Initialize the DatasetComposer.

        Parameters
        ----------
        generator_proportions : Dict[str, float], optional
            Dictionary mapping generator names to their proportions in the dataset.
            The proportions should sum to 1.0. If None, uses default proportions.
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        """
        self.global_seed = global_seed
        self.rng = np.random.default_rng(global_seed)

        # Setup generator proportions
        self._setup_proportions(generator_proportions)

        # Create generator wrappers
        self.generator_wrappers = self._create_generator_wrappers()

        # Validate proportions
        self._validate_proportions()

        # Set random seed
        torch.manual_seed(self.global_seed)

    def _setup_proportions(self, generator_proportions):
        """Setup default or custom generator proportions."""
        default_generator_proportions = {
            "forecast_pfn": 0.04,
            "gp": 0.27,
            "kernel": 0.24,
            "lmc": 0.43,
            "sine_wave": 0.02,
        }
        self.generator_proportions = (
            generator_proportions or default_generator_proportions
        )

    def _create_generator_wrappers(self) -> Dict[GeneratorWrapper, float]:
        """
        Create generator wrappers based on the specified proportions.

        Returns:
            Dictionary mapping generator wrappers to their proportions.
        """
        generator_map = {
            "lmc": (LMCGeneratorWrapper, LMCGeneratorParams),
            "kernel": (KernelGeneratorWrapper, KernelGeneratorParams),
            "gp": (GPGeneratorWrapper, GPGeneratorParams),
            "forecast_pfn": (
                ForecastPFNGeneratorWrapper,
                ForecastPFNGeneratorParams,
            ),
            "sine_wave": (SineWaveGeneratorWrapper, SineWaveGeneratorParams),
        }

        wrappers = {}
        # Create the base parameter instance
        base_params = GeneratorParams(global_seed=self.global_seed)

        for name, (wrapper_class, gen_param_class) in generator_map.items():
            if name in self.generator_proportions:
                try:
                    # Create generator-specific parameters with defaults
                    gen_params = gen_param_class(global_seed=self.global_seed)

                    # Override with base parameters
                    gen_params.total_length = base_params.total_length
                    gen_params.future_length = base_params.future_length
                    gen_params.num_channels = base_params.num_channels
                    gen_params.global_seed = self.global_seed
                    gen_params.distribution_type = base_params.distribution_type

                    # Re-run validation after parameter updates
                    gen_params.__post_init__()

                    # Create the wrapper with the properly configured parameters
                    wrappers[wrapper_class(gen_params)] = self.generator_proportions[
                        name
                    ]

                except Exception as e:
                    logger.warning(f"Failed to create {name} generator: {e}")
                    continue

        if not wrappers:
            raise ValueError("No valid generators created")

        # Normalize proportions
        total_prop = sum(wrappers.values())
        if total_prop == 0:
            raise ValueError("Total generator proportions cannot be zero")

        return {wrapper: prop / total_prop for wrapper, prop in wrappers.items()}

    def _validate_proportions(self) -> None:
        """
        Validate that the generator proportions sum to approximately 1.0.
        """
        total_proportion = sum(self.generator_wrappers.values())
        if not np.isclose(total_proportion, 1.0, atol=1e-6):
            raise ValueError(
                f"Generator proportions should sum to 1.0, got {total_proportion}"
            )

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> Tuple[BatchTimeSeriesContainer, str]:
        """
        Generate a single batch by randomly selecting a generator according to proportions.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate per batch.
        seed : int, optional
            Random seed for reproducibility (default: None).

        Returns
        -------
        Tuple[BatchTimeSeriesContainer, str]
            Tuple containing (batch_data, generator_name).
        """
        # Set seed if provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Select generator based on proportions
        generators = list(self.generator_wrappers.keys())
        proportions = list(self.generator_wrappers.values())
        selected_generator = self.rng.choice(generators, p=proportions)

        # Generate batch
        batch = selected_generator.generate_batch(
            batch_size=batch_size,
            seed=seed,
        )

        return batch, selected_generator.__class__.__name__

    def generate_dataset(
        self,
        num_batches: int,
        batch_size: int,
        save_path: Optional[str] = None,
    ) -> List[BatchTimeSeriesContainer]:
        """
        Generate a dataset with the specified number of batches.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series to generate per batch.
        save_path : str, optional
            Path to save the dataset (default: None).

        Returns
        -------
        List[BatchTimeSeriesContainer]
            List of batch data.
        """
        batches = []
        stats: Dict[str, int] = {}

        logger.info(f"Generating {num_batches} batches...")
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            seed = self.global_seed + batch_idx
            batch, generator_info = self.generate_batch(
                batch_size=batch_size,
                seed=seed,
            )
            batches.append(batch)
            stats[generator_info] = stats.get(generator_info, 0) + 1

            if save_path:
                self._save_batch(batch, batch_idx, save_path)

        logger.info("Dataset generation completed")
        logger.info(f"Statistics: {stats}")

        return batches

    def _save_batch(
        self,
        batch: BatchTimeSeriesContainer,
        batch_idx: int,
        output_dir: str,
    ) -> None:
        """
        Save a batch to disk.

        Parameters
        ----------
        batch : BatchTimeSeriesContainer
            The batch to save.
        batch_idx : int
            Batch index.
        output_dir : str
            Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)
        batch_path = os.path.join(output_dir, f"batch_{batch_idx:05d}.pt")
        torch.save(batch, batch_path)
        logger.debug(f"Saved batch {batch_idx} to {batch_path}")

    def save_dataset(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        save_as_single_file: bool = False,
    ) -> None:
        """
        Generate and save a dataset to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save the dataset.
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        save_as_single_file : bool, optional
            If True, save all batches in a single file (default: False).
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving dataset to {output_dir} with {num_batches} batches")

        # Generate the dataset
        batches = self.generate_dataset(
            num_batches=num_batches,
            batch_size=batch_size,
            save_path=None if save_as_single_file else output_dir,
        )

        # Save as a single file if requested
        if save_as_single_file:
            combined_file_path = os.path.join(output_dir, "dataset.pt")
            torch.save(batches, combined_file_path)
            logger.info(f"Saved combined dataset to {combined_file_path}")

        logger.info(f"Dataset saved to {output_dir}")

    def create_train_validation_datasets(
        self,
        output_dir: str,
        train_batches: int,
        val_batches: int,
        batch_size: int,
        save_as_single_file: bool = False,
    ) -> None:
        """
        Create and save both training and validation datasets.

        Parameters
        ----------
        output_dir : str
            Directory to save the datasets.
        train_batches : int
            Number of batches for the training dataset.
        val_batches : int
            Number of batches for the validation dataset.
        batch_size : int
            Number of time series per batch.
        save_as_single_file : bool, optional
            If True, save all batches in a single file per dataset (default: False).
        """
        # Create output directories
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        logger.info(f"Creating training dataset with {train_batches} batches")
        self.save_dataset(
            output_dir=train_dir,
            num_batches=train_batches,
            batch_size=batch_size,
            save_as_single_file=save_as_single_file,
        )

        # Use a different seed for validation
        val_seed = self.global_seed + 10000
        original_seed = self.global_seed
        self.global_seed = val_seed

        logger.info(f"Creating validation dataset with {val_batches} batches")
        self.save_dataset(
            output_dir=val_dir,
            num_batches=val_batches,
            batch_size=batch_size,
            save_as_single_file=save_as_single_file,
        )

        # Restore original seed
        self.global_seed = original_seed

        logger.info(f"Training and validation datasets saved to {output_dir}")


class OnTheFlyDatasetGenerator:
    """
    Provides an interface for on-the-fly generation of batches during training.
    """

    def __init__(
        self,
        composer: DatasetComposer,
        batch_size: int,
        buffer_size: int = 10,
        global_seed: int = 42,
    ):
        """
        Initialize the OnTheFlyDatasetGenerator.

        Parameters
        ----------
        composer : DatasetComposer
            The dataset composer to use for generation.
        batch_size : int
            Number of time series per batch.
        buffer_size : int, optional
            Number of batches to pre-generate and keep in buffer (default: 10).
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        """
        self.composer = composer
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.global_seed = global_seed

        # Initialize batch buffer
        self.buffer = []
        self.batch_counter = 0

        # Fill the buffer initially
        self._fill_buffer()

    def _fill_buffer(self) -> None:
        """
        Fill the batch buffer with pre-generated batches.
        """
        while len(self.buffer) < self.buffer_size:
            seed = self.global_seed + self.batch_counter
            batch, _ = self.composer.generate_batch(
                batch_size=self.batch_size,
                seed=seed,
            )
            self.buffer.append(batch)
            self.batch_counter += 1

    def get_batch(self) -> BatchTimeSeriesContainer:
        """
        Get a batch from the buffer and refill as needed.

        Returns
        -------
        BatchTimeSeriesContainer
            A batch of time series data.
        """
        # Get a batch from the buffer
        batch = self.buffer.pop(0)

        # Generate a new batch to refill the buffer
        seed = self.global_seed + self.batch_counter
        new_batch, _ = self.composer.generate_batch(
            batch_size=self.batch_size,
            seed=seed,
        )
        self.buffer.append(new_batch)
        self.batch_counter += 1

        return batch
