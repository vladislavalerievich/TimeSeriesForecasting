import argparse
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
    """Manages Arrow dataset files for time series data."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

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

    def get_current_count(self) -> int:
        """Get the current number of series in the dataset."""
        if not self.output_path.exists():
            return 0

        try:
            table = pq.read_table(self.output_path)
            return len(table)
        except Exception as e:
            logging.warning(f"Error reading existing file: {e}. Starting from 0.")
            return 0

    def append_batch(self, batch_data: List[Dict[str, Any]]) -> None:
        """Append a batch of time series to the dataset."""
        if not batch_data:
            return

        # Convert batch to Arrow Table directly to avoid pandas conversion issues
        arrays = []
        for field in self.schema:
            field_name = field.name
            if field_name in ["start", "generation_timestamp"]:
                # Convert timestamps to proper Arrow format
                timestamps = [row[field_name] for row in batch_data]
                # Convert to Arrow timestamp array
                arrays.append(
                    pa.array([ts.value for ts in timestamps], type=pa.timestamp("ns"))
                )
            else:
                arrays.append(pa.array([row[field_name] for row in batch_data]))

        new_table = pa.Table.from_arrays(arrays, schema=self.schema)

        if self.output_path.exists():
            # Read existing table and concatenate
            existing_table = pq.read_table(self.output_path)
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table

        # Write back to file
        pq.write_table(combined_table, self.output_path)


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


class ContinuousGenerator:
    """Main class for continuous time series generation."""

    def __init__(
        self,
        generator_wrapper: GeneratorWrapper,
        dataset_manager: TimeSeriesDatasetManager,
        batch_size: int = 100,
        log_interval: int = 1000,
    ):
        self.generator_wrapper = generator_wrapper
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.log_interval = log_interval

        # Initialize counter from existing dataset
        self.current_count = dataset_manager.get_current_count()
        logging.info(f"Starting from series count: {self.current_count}")

    def generate_batch(self, start_id: int) -> List[Dict[str, Any]]:
        """Generate a batch of time series."""
        batch_data = []

        for i in range(self.batch_size):
            series_id = start_id + i

            # Use series_id as seed for reproducibility
            series_data = self.generator_wrapper.generate_series(random_seed=series_id)

            # Ensure values are in proper format
            values = self.generator_wrapper._ensure_proper_format(series_data["values"])

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
        dataset_manager = TimeSeriesDatasetManager(str(output_path))

        # Create continuous generator
        continuous_gen = ContinuousGenerator(
            generator_wrapper=generator_wrapper,
            dataset_manager=dataset_manager,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
        )

        # Run continuous generation
        continuous_gen.run_continuous(max_series=args.max_series)

        logging.info("Generation completed successfully!")

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
