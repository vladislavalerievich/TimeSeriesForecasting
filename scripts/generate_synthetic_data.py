import os

from src.synthetic_generation.data_loaders import (
    SyntheticTrainDataLoader,
    SyntheticValidationDataLoader,
)
from src.synthetic_generation.dataset_composer import (
    DatasetComposer,
    OnTheFlyDatasetGenerator,
)
from src.synthetic_generation.kernel_generator_wrapper import (
    KernelGeneratorParams,
    KernelGeneratorWrapper,
)
from src.synthetic_generation.lmc_generator_wrapper import (
    LMCGeneratorParams,
    LMCGeneratorWrapper,
)

if __name__ == "__main__":
    global_seed = 42

    # Set up parameters
    lmc_params = LMCGeneratorParams(
        global_seed=global_seed,
        history_length=256,
        target_length=64,
        num_channels=[1, 8],
    )
    kernel_params = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=256,
        target_length=64,
        num_channels=[1, 8],
    )

    lmc_gen = LMCGeneratorWrapper(lmc_params)
    kernel_gen = KernelGeneratorWrapper(kernel_params)

    generator_proportions = {lmc_gen: 0.8, kernel_gen: 0.2}
    composer = DatasetComposer(
        generator_proportions=generator_proportions, global_seed=global_seed
    )

    save_dir = "data/synthetic_val_data_lmc_80_kernel_20_batches_10_batch_size_64"
    os.makedirs(save_dir, exist_ok=True)

    print("--- Generating and saving 10 batches to disk ---")
    composer.save_dataset(output_dir=save_dir, num_batches=10, batch_size=64)

    # Load and print shapes from saved dataset
    val_loader = SyntheticValidationDataLoader(data_path=save_dir)
    for i, batch in enumerate(val_loader):
        print(
            f"[Saved] Batch {i}: history_values {batch.history_values.shape}, target_values {batch.target_values.shape}"
        )

    print("\n--- Generating 4 batches on the fly ---")
    on_the_fly_gen = OnTheFlyDatasetGenerator(
        composer=composer, batch_size=2, buffer_size=2, global_seed=global_seed
    )
    train_loader = SyntheticTrainDataLoader(
        generator=on_the_fly_gen, num_batches_per_epoch=4
    )
    for i, batch in enumerate(train_loader):
        print(
            f"[On-the-fly] Batch {i}: history_values {batch.history_values.shape}, target_values {batch.target_values.shape}"
        )
        if i >= 3:
            break
