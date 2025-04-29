from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)

if __name__ == "__main__":
    # Create a generator
    generator = MultivariateTimeSeriesGenerator(global_seed=42)

    # Generate a single batch
    batch = generator.generate_batch(
        batch_size=64, history_length=128, target_length=64, num_channels=10
    )
    print("Batch shape:", batch.shape)  # Should be (64, 10, 128 + 64)
    print("Batch dtype:", batch.dtype)  # Should be torch.float32
    print("Batch device:", batch.device)  # Should be the same as the generator's device

    # Generate and save a full dataset
    generator.save_dataset(
        output_dir="synthetic_dataset",
        num_batches=100,
        batch_size=64,
        min_history_length=64,
        max_history_length=256,
        min_target_length=32,
        max_target_length=128,
        fixed_num_channels=160,  # Fixed channel count
        n_jobs=8,  # Use 8 CPU cores
    )

    # Use with PyTorch DataLoader
    # dataset = MultivariateTimeSeriesDataset(
    #     num_batches=100, batch_size=64, min_history_length=64, max_history_length=256
    # )
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
