from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)

if __name__ == "__main__":
    # Create a generator
    generator = MultivariateTimeSeriesGenerator(global_seed=42)

    # Generate and save a full dataset
    generator.save_dataset(
        output_dir="outputs/datasets/mts",
        num_batches=100,
        batch_size=64,
        save_as_single_file=False,
        num_cpus=12,
    )

    print("Dataset saved successfully.")
