import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.synthetic_generation.common.utils import (
    select_safe_random_frequency,
    select_safe_start_date,
)
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator import (
    ForecastPFNGenerator,
)
from src.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    SineWaveGeneratorParams,
)
from src.synthetic_generation.gp_prior.gp_generator import GPGenerator
from src.synthetic_generation.kernel_synth.kernel_synth import KernelSynthGenerator
from src.synthetic_generation.sine_waves.sine_wave_generator import SineWaveGenerator


class TimeSeriesDatasetManager:
    """Manages Arrow dataset files for time series data with memory-efficient batching."""

    def __init__(self, output_path: str, combine_every: int = 1000):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create batches directory
        self.batches_dir = self.output_path.parent / "batches"
        self.batches_dir.mkdir(exist_ok=True)

        # Metadata file to track state
        self.metadata_path = self.output_path.parent / "metadata.json"

        # How often to combine batches into main dataset
        self.combine_every = combine_every

        # Current batch counter
        self.batch_counter = 0
        self.last_combined_batch = -1

        # Define schema for time series data
        self.schema = pa.schema(
            [
                ("series_id", pa.int64()),
                ("values", pa.list_(pa.float64())),
                ("length", pa.int32()),
                ("generator_type", pa.string()),
                ("start", pa.timestamp("ns")),  # Start time of the time series
                ("frequency", pa.string()),  # Frequency of the time series
                ("generation_timestamp", pa.timestamp("ns")),  # When it was generated
            ]
        )

        # Load existing state if available
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata about current state."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                self.batch_counter = metadata.get("batch_counter", 0)
                self.last_combined_batch = metadata.get("last_combined_batch", -1)
                logging.info(
                    f"Loaded metadata: batch_counter={self.batch_counter}, last_combined_batch={self.last_combined_batch}"
                )
            except Exception as e:
                logging.warning(f"Error loading metadata: {e}. Starting fresh.")
                self.batch_counter = 0
                self.last_combined_batch = -1

    def _save_metadata(self) -> None:
        """Save current state metadata."""
        try:
            metadata = {
                "batch_counter": self.batch_counter,
                "last_combined_batch": self.last_combined_batch,
                "output_path": str(self.output_path),
                "combine_every": self.combine_every,
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving metadata: {e}")

    def get_current_count(self) -> int:
        """Get the current number of series in the dataset efficiently."""
        total_count = 0

        # Count from main dataset file if it exists
        if self.output_path.exists():
            try:
                # Use PyArrow's metadata to get row count without loading data
                parquet_file = pq.ParquetFile(self.output_path)
                total_count += parquet_file.metadata.num_rows
            except Exception as e:
                logging.warning(f"Error reading main dataset metadata: {e}")

        # Count from batch files
        batch_files = sorted(self.batches_dir.glob("batch_*.parquet"))
        for batch_file in batch_files:
            try:
                parquet_file = pq.ParquetFile(batch_file)
                total_count += parquet_file.metadata.num_rows
            except Exception as e:
                logging.warning(f"Error reading batch file {batch_file}: {e}")

        return total_count

    def append_batch(self, batch_data: List[Dict[str, Any]]) -> None:
        """Append a batch of time series to the dataset using batch files."""
        if not batch_data:
            return

        try:
            # Convert batch to Arrow Table
            arrays = []
            for field in self.schema:
                field_name = field.name
                if field_name in ["start", "generation_timestamp"]:
                    # Convert timestamps to proper Arrow format
                    timestamps = [row[field_name] for row in batch_data]
                    # Convert to Arrow timestamp array
                    arrays.append(
                        pa.array(
                            [ts.value for ts in timestamps], type=pa.timestamp("ns")
                        )
                    )
                else:
                    arrays.append(pa.array([row[field_name] for row in batch_data]))

            new_table = pa.Table.from_arrays(arrays, schema=self.schema)

            # Write batch to individual file
            batch_filename = f"batch_{self.batch_counter:08d}.parquet"
            batch_filepath = self.batches_dir / batch_filename
            pq.write_table(new_table, batch_filepath)

            self.batch_counter += 1

            # Check if we should combine batches
            if self.batch_counter % self.combine_every == 0:
                self._combine_batches()

            # Save metadata
            self._save_metadata()

        except Exception as e:
            logging.error(f"Error writing batch: {e}")
            raise

    def _combine_batches(self) -> None:
        """Combine all batch files into the main dataset."""
        try:
            # Get all batch files that haven't been combined yet
            all_batch_files = sorted(self.batches_dir.glob("batch_*.parquet"))

            # Filter to only files after last_combined_batch
            batch_files = [
                f
                for f in all_batch_files
                if self._extract_batch_number(f) > self.last_combined_batch
            ]

            if not batch_files:
                logging.info("No new batch files to combine")
                return

            logging.info(
                f"Combining {len(batch_files)} batch files into {self.output_path}"
            )

            # Read all batch files
            tables = []

            # Add existing main dataset if it exists
            if self.output_path.exists():
                main_table = pq.read_table(self.output_path)
                tables.append(main_table)

            # Read batch files
            for batch_file in batch_files:
                batch_table = pq.read_table(batch_file)
                tables.append(batch_table)

            # Combine all tables
            if tables:
                combined_table = pa.concat_tables(tables)

                # Write combined dataset
                pq.write_table(combined_table, self.output_path)

                # Update last combined batch
                self.last_combined_batch = max(
                    self._extract_batch_number(f) for f in batch_files
                )

                # Clean up batch files
                for batch_file in batch_files:
                    try:
                        batch_file.unlink()
                    except Exception as e:
                        logging.warning(f"Error deleting batch file {batch_file}: {e}")

                logging.info(
                    f"Combined dataset written to {self.output_path} with {len(combined_table)} rows"
                )
                logging.info(
                    f"Batch combination completed in {len(batch_files)} files processed"
                )

        except Exception as e:
            logging.error(f"Error combining batches: {e}")
            # Don't raise here to allow generation to continue

    def _extract_batch_number(self, batch_file: Path) -> int:
        """Extract batch number from filename."""
        try:
            # Extract number from filename like "batch_00000123.parquet"
            name = batch_file.stem
            return int(name.split("_")[1])
        except:
            return -1

    def finalize_dataset(self) -> None:
        """Final combination of any remaining batch files."""
        logging.info("Finalizing dataset by combining batch files...")

        # Get all remaining batch files
        batch_files = sorted(self.batches_dir.glob("batch_*.parquet"))

        if not batch_files:
            logging.info("No batch files found to combine")
            return

        try:
            # Force combine all remaining batches
            old_combine_every = self.combine_every
            self.combine_every = 1  # Force combination
            self._combine_batches()
            self.combine_every = old_combine_every

            logging.info("Dataset finalization complete")

        except Exception as e:
            logging.error(f"Error during finalization: {e}")

        # Save final metadata
        self._save_metadata()

    def close(self) -> None:
        """Clean up and close the dataset manager."""
        try:
            self.finalize_dataset()
            logging.info("Dataset file closed successfully.")
        except Exception as e:
            logging.error(f"Error closing dataset: {e}")


class GeneratorWrapper:
    """Unified wrapper for different generator types."""

    def __init__(self, generator_type: str, length: int = 2048, **kwargs):
        self.generator_type = generator_type
        self.length = length
        self.rng = np.random.default_rng(kwargs.get("global_seed", 42))

        if generator_type.lower() == "gp":
            params = GPGeneratorParams(total_length=length, **kwargs)
            self.generator = GPGenerator(params, length=length)
        elif generator_type.lower() == "kernel":
            params = KernelGeneratorParams(total_length=length, **kwargs)
            self.generator = KernelSynthGenerator(length=length)
        elif generator_type.lower() == "forecastpfn":
            params = ForecastPFNGeneratorParams(total_length=length, **kwargs)
            self.generator = ForecastPFNGenerator(params, length=length)
        elif generator_type.lower() == "sinewave":
            params = SineWaveGeneratorParams(total_length=length, **kwargs)
            self.generator = SineWaveGenerator(
                length=length,
                period_range=params.period_range,
                amplitude_range=params.amplitude_range,
                phase_range=params.phase_range,
                noise_level=params.noise_level,
            )
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")

    def generate_series(self, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a single time series and return all relevant information."""
        try:
            # Generate frequency and start datetime for ALL generators
            frequency = select_safe_random_frequency(self.length, self.rng)
            start = select_safe_start_date(
                total_length=self.length,
                frequency=frequency,
                rng=self.rng,
            )

            if self.generator_type.lower() == "sinewave":
                # Generate diverse sine wave with randomized parameters for realism
                if random_seed is not None:
                    local_rng = np.random.default_rng(random_seed)
                else:
                    local_rng = self.rng

                # Randomize parameters for diverse, realistic sine waves
                # Period range: vary from short-term (5-20) to medium-term (20-200) patterns
                period_min = local_rng.uniform(5.0, 15.0)
                period_max = local_rng.uniform(
                    period_min + 5.0, min(period_min + 185.0, self.length / 3)
                )
                period_range = (period_min, period_max)

                # Amplitude: vary from small (0.1) to large (5.0) for different signal strengths
                amp_min = local_rng.uniform(0.1, 1.0)
                amp_max = local_rng.uniform(amp_min + 0.2, amp_min + 4.0)
                amplitude_range = (amp_min, amp_max)

                # Phase: full random phase for temporal diversity
                phase_range = (0.0, 2.0 * np.pi)

                # Noise level: vary from 0.0 to 0.1 uniformly as requested
                noise_level = local_rng.uniform(0.0, 0.1)

                # Create sine wave generator with randomized parameters
                sine_generator = SineWaveGenerator(
                    length=self.length,
                    period_range=period_range,
                    amplitude_range=amplitude_range,
                    phase_range=phase_range,
                    noise_level=noise_level,
                    random_seed=random_seed,
                )

                result = sine_generator.generate_time_series(random_seed=random_seed)

                return {
                    "values": result,
                    "start": pd.Timestamp(start),
                    "frequency": frequency.value,  # Convert enum to string
                }
            elif self.generator_type.lower() == "forecastpfn":
                result = self.generator.generate_time_series(
                    start=start, random_seed=random_seed, periodicity=frequency
                )

                return {
                    "values": result["values"],  # Extract just the values array
                    "start": pd.Timestamp(start),
                    "frequency": frequency.value,  # Convert enum to string
                }
            else:
                result = self.generator.generate_time_series(random_seed=random_seed)

                return {
                    "values": result,
                    "start": pd.Timestamp(start),
                    "frequency": frequency.value,  # Convert enum to string
                }

        except Exception as e:
            logging.error(f"Error generating series: {e}")

    def _ensure_proper_format(self, values: Any) -> np.ndarray:
        """Ensure values are in proper numpy array format with correct length."""
        # Ensure numpy array and proper shape
        values = np.asarray(values)
        if values.ndim > 1:
            values = values.flatten()

        # Ensure correct length
        if len(values) != self.length:
            logging.warning(
                f"Generated series length {len(values)} != expected {self.length}"
            )
            if len(values) > self.length:
                values = values[: self.length]
            else:
                # Pad with zeros if too short
                values = np.pad(values, (0, self.length - len(values)), mode="constant")

        return values.astype(np.float64)


def generate_one_for_batch(series_id, generator_type, length, global_seed):
    # Re-create the generator wrapper in each process
    generator_wrapper = GeneratorWrapper(
        generator_type=generator_type,
        length=length,
        global_seed=global_seed,
    )
    series_data = generator_wrapper.generate_series(random_seed=series_id)
    values = generator_wrapper._ensure_proper_format(series_data["values"])
    return {
        "series_id": series_id,
        "values": values.tolist(),
        "length": len(values),
        "generator_type": generator_type,
        "start": series_data["start"],
        "frequency": series_data["frequency"],
        "generation_timestamp": pd.Timestamp.now(),
    }


class ContinuousGenerator:
    """Main class for continuous time series generation."""

    def __init__(
        self,
        generator_wrapper: GeneratorWrapper,
        dataset_manager: TimeSeriesDatasetManager,
        batch_size: int = 100,
        log_interval: int = 100,
        num_workers: int = 16,
    ):
        self.generator_wrapper = generator_wrapper
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.num_workers = num_workers

        # Initialize counter from existing dataset
        self.current_count = dataset_manager.get_current_count()
        logging.info(f"Starting from series count: {self.current_count}")

    def generate_batch(self, start_id: int) -> List[Dict[str, Any]]:
        """Generate a batch of time series, using multiprocessing if num_workers > 1."""
        batch_data = []
        series_ids = [start_id + i for i in range(self.batch_size)]

        if self.num_workers > 1:
            import concurrent.futures

            # Use top-level function for multiprocessing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                func = functools.partial(
                    generate_one_for_batch,
                    generator_type=self.generator_wrapper.generator_type,
                    length=self.generator_wrapper.length,
                    global_seed=getattr(
                        self.generator_wrapper, "rng", 42
                    ).bit_generator._seed_seq.entropy
                    if hasattr(self.generator_wrapper, "rng")
                    else 42,
                )
                batch_data = list(executor.map(func, series_ids))
        else:
            for series_id in series_ids:
                series_data = self.generator_wrapper.generate_series(
                    random_seed=series_id
                )
                values = self.generator_wrapper._ensure_proper_format(
                    series_data["values"]
                )
                batch_data.append(
                    {
                        "series_id": series_id,
                        "values": values.tolist(),  # Convert to list for Arrow
                        "length": len(values),
                        "generator_type": self.generator_wrapper.generator_type,
                        "start": series_data["start"],
                        "frequency": series_data["frequency"],
                        "generation_timestamp": pd.Timestamp.now(),
                    }
                )
        return batch_data

    def run_continuous(self, max_series: Optional[int] = None) -> None:
        """Run continuous generation until interrupted."""
        logging.info(f"Starting continuous generation from series {self.current_count}")
        logging.info(
            f"Batch size: {self.batch_size}, Log interval: {self.log_interval}"
        )

        if max_series:
            logging.info(f"Will generate up to {max_series} total series")

        start_time = time.time()
        last_log_time = start_time

        try:
            while True:
                # Check if we've reached the maximum
                if max_series and self.current_count >= max_series:
                    logging.info(f"Reached maximum series limit: {max_series}")
                    break

                # Generate batch
                batch_start_time = time.time()
                batch_data = self.generate_batch(self.current_count)
                generation_time = time.time() - batch_start_time

                # Write batch to dataset
                write_start_time = time.time()

                # Check if this will trigger a batch combination
                will_combine = (
                    self.dataset_manager.batch_counter + 1
                ) % self.dataset_manager.combine_every == 0
                if will_combine:
                    logging.info(
                        f"Will combine batches every {self.dataset_manager.combine_every} series..."
                    )

                self.dataset_manager.append_batch(batch_data)
                write_time = time.time() - write_start_time

                # Update counter
                self.current_count += self.batch_size

                # Log progress
                if (
                    self.current_count % self.log_interval == 0
                    or time.time() - last_log_time > 60
                ):
                    elapsed = time.time() - start_time
                    series_per_sec = self.current_count / elapsed if elapsed > 0 else 0

                    logging.info(
                        f"Generated {self.current_count} series | "
                        f"Rate: {series_per_sec:.1f} series/sec | "
                        f"Batch gen: {generation_time:.2f}s, write: {write_time:.2f}s"
                    )
                    last_log_time = time.time()

        except KeyboardInterrupt:
            logging.info(
                f"Interrupted by user. Generated {self.current_count} series total."
            )
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            raise
        finally:
            # Always finalize the dataset when generation stops
            self.dataset_manager.finalize_dataset()
            self.dataset_manager.close()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous time series generation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--generator",
        type=str,
        required=True,
        choices=["gp", "kernel", "forecastpfn", "sinewave"],
        help="Type of generator to use",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/work/dlclarge2/moroshav-GiftEvalPretrain/SynthDatasets2048/",
        help="Base output directory for datasets",
    )

    # Optional arguments
    parser.add_argument(
        "--length", type=int, default=2048, help="Length of each time series"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of series to generate before writing to disk",
    )

    parser.add_argument(
        "--log-interval", type=int, default=1000, help="Log progress every N series"
    )

    parser.add_argument(
        "--combine-every",
        type=int,
        default=1000,
        help="Combine batch files into main dataset every N batches",
    )

    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Maximum number of series to generate (default: unlimited)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Generator-specific parameters
    parser.add_argument(
        "--global-seed", type=int, default=42, help="Global random seed"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel generation (default: 1)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    output_path = Path(args.output_dir) / args.generator / "dataset.arrow"

    try:
        # Initialize components
        logging.info(
            f"Initializing {args.generator} generator with length {args.length}"
        )
        generator_wrapper = GeneratorWrapper(
            generator_type=args.generator,
            length=args.length,
            global_seed=args.global_seed,
        )

        logging.info(f"Setting up dataset at {output_path}")
        dataset_manager = TimeSeriesDatasetManager(
            str(output_path), combine_every=args.combine_every
        )

        # Create continuous generator
        continuous_gen = ContinuousGenerator(
            generator_wrapper=generator_wrapper,
            dataset_manager=dataset_manager,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
            num_workers=args.num_workers,
        )

        # Run continuous generation
        continuous_gen.run_continuous(max_series=args.max_series)

        logging.info("Generation completed successfully!")

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
