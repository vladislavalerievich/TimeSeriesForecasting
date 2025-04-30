from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)

if __name__ == "__main__":
    # Create a generator
    generator = MultivariateTimeSeriesGenerator(global_seed=42)

    # Generate a single batch
    # batch = generator.generate_batch(
    #     batch_size=32, history_length=128, target_length=64, num_channels=10
    # )
    # print("Batch history_values shape:", batch.history_values.shape)
    # print("Batch target_values shape:", batch.target_values.shape)
    # print("Batch target_channels_indices shape:", batch.target_channels_indices.shape)
    # print(
    #     "Batch history_values dtype:", batch.history_values.dtype
    # )  # Should be torch.float32
    # print(
    #     "Batch target_channels_indices dtype:", batch.target_channels_indices.dtype
    # )  # Should be torch.int64
    # print("Batch history_time_features shape:", batch.history_time_features.shape)
    # print("Batch target_time_features shape:", batch.target_time_features.shape)
    # print("Batch target_values device:", batch.target_values.device)

    # Generate and save a full dataset
    generator.save_dataset(
        output_dir="outputs/datasets/mts",
        num_batches=10,
        batch_size=5,
        save_as_single_file=True,
        num_cpus=1,
    )

    print("Dataset saved successfully.")
