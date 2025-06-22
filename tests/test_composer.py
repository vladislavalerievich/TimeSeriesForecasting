import logging
from collections import Counter, defaultdict

from src.synthetic_generation.dataset_composer import DefaultSyntheticComposer

# Suppress verbose logging from the generator
logging.basicConfig(level=logging.WARNING)


def test_synthetic_composer():
    """
    Tests the DefaultSyntheticComposer by generating batches and inspecting their properties.

    This test verifies that the hierarchical composition works as expected by:
    1. Printing the configured generator proportions for each forecast range.
    2. Generating 100 batches and tracking their source and shapes.
    3. Printing a summary of the actual generated batches, including counts and
       the unique (history, target, channels) dimensions for each generator.
    """
    print("--- Initializing DefaultSyntheticComposer ---")
    composer = DefaultSyntheticComposer(seed=42)

    # --- 1. Print Configured Proportions ---
    print("\n--- Configured Proportions ---")
    print(f"Range Proportions: {composer.composer.range_proportions}")

    for range_name, gen_props in composer.generator_proportions.items():
        print(f"\n[{range_name.capitalize()} Range]")
        for gen_name, prop in gen_props.items():
            print(f"  - {gen_name}: {prop:.2f}")

    # --- 2. Generate Batches and Collect Stats ---
    num_batches = 100
    batch_size = 8
    generator_counts = Counter()
    # A dictionary where each value is a set of unique shape tuples
    generator_shapes = defaultdict(set)

    print(f"\n--- Generating {num_batches} batches of size {batch_size} ---")
    for i in range(num_batches):
        batch, generator_full_name = composer.generate_batch(
            batch_size=batch_size, seed=42 + i
        )
        generator_counts[generator_full_name] += 1

        # Extract shape information
        history_len = batch.history_values.shape[1]
        target_len = batch.future_values.shape[1]
        num_channels = batch.history_values.shape[2]
        shape_tuple = (history_len, target_len, num_channels)

        generator_shapes[generator_full_name].add(shape_tuple)
    print("Generation complete.")

    # --- 3. Process and Print Generation Results ---
    print("\n--- Generation Results ---")

    # Group results by range for structured printing
    results_by_range = defaultdict(list)
    for gen_full_name, count in generator_counts.items():
        parts = gen_full_name.split("_")
        range_name = parts[-1]
        base_gen_name = "_".join(parts[:-1])
        results_by_range[range_name].append(
            {
                "name": base_gen_name,
                "count": count,
                "shapes": sorted(list(generator_shapes[gen_full_name])),
            }
        )

    for range_name in sorted(results_by_range.keys()):
        print(
            f"\n[{range_name.capitalize()} Range] - Generated {sum(item['count'] for item in results_by_range[range_name])} batches"
        )
        for result in sorted(results_by_range[range_name], key=lambda x: x["name"]):
            print(f"  -> {result['name']} (Generated {result['count']} batches)")
            print("     Unique (history, target, channels) shapes:")
            for shape in result["shapes"]:
                print(f"       - {shape}")


if __name__ == "__main__":
    test_synthetic_composer()
