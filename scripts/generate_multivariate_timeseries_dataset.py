from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)

if __name__ == "__main__":
    # Create a generator
    generator = MultivariateTimeSeriesGenerator(
        global_seed=42,
        history_length=96,
        target_length=96,
        max_target_channels=2,
        num_channels=160,
    )

    # Generate and save a full dataset
    generator.save_dataset(
        output_dir="outputs/datasets/mts",
        num_batches=100,
        batch_size=64,
        save_as_single_file=True,
        num_cpus=16,
    )

    print("Dataset saved successfully.")
