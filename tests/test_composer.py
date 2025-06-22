import logging
import time
from collections import Counter, defaultdict

from src.synthetic_generation.dataset_composer import DefaultSyntheticComposer

# Configure logging to show timing warnings
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_synthetic_composer():
    """
    Tests the DefaultSyntheticComposer by generating batches and inspecting their properties.

    This test verifies that the hierarchical composition works as expected by:
    1. Printing the configured generator proportions for each forecast range.
    2. Generating 100 batches and tracking their source and shapes.
    3. Printing a summary of the actual generated batches, including counts and
       the unique (history, future, channels) dimensions for each generator.
    """
    print("--- Initializing DefaultSyntheticComposer ---")
    init_start = time.time()
    composer = DefaultSyntheticComposer(seed=42)
    init_time = time.time() - init_start
    print(f"Initialization took {init_time:.4f}s")

    # --- 1. Print Configured Proportions ---
    print("\n--- Configured Proportions ---")
    print(f"Range Proportions: {composer.composer.range_proportions}")

    for range_name, gen_props in composer.generator_proportions.items():
        print(f"\n[{range_name.capitalize()} Range]")
        for gen_name, prop in gen_props.items():
            print(f"  - {gen_name}: {prop:.2f}")

    # --- 2. Generate Batches and Collect Stats ---
    num_batches = 100
    batch_size = 64
    generator_counts = Counter()
    # A dictionary where each value is a set of unique shape tuples
    generator_shapes = defaultdict(set)
    batch_times = []

    print(f"\n--- Generating {num_batches} batches of size {batch_size} ---")
    generation_start = time.time()

    for i in range(num_batches):
        batch_start = time.time()
        print(f"Generating batch {i + 1}/{num_batches}...")

        batch, generator_full_name = composer.generate_batch(
            batch_size=batch_size, seed=42 + i
        )

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(
            f"  -> Batch {i + 1} of history length {batch.history_values.shape[1]} and future length {batch.future_values.shape[1]} completed in {batch_time:.4f}s using {generator_full_name}"
        )

        generator_counts[generator_full_name] += 1

        # Extract shape information
        history_len = batch.history_values.shape[1]
        future_len = batch.future_values.shape[1]
        num_channels = batch.history_values.shape[2]
        shape_tuple = (history_len, future_len, num_channels)

        generator_shapes[generator_full_name].add(shape_tuple)

    total_generation_time = time.time() - generation_start
    print(f"Total generation took {total_generation_time:.4f}s")
    print(f"Average batch time: {sum(batch_times) / len(batch_times):.4f}s")
    print(f"Min batch time: {min(batch_times):.4f}s")
    print(f"Max batch time: {max(batch_times):.4f}s")

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
            print("     Unique (history, future, channels) shapes:")
            for shape in result["shapes"]:
                print(f"       - {shape}")


if __name__ == "__main__":
    overall_start = time.time()
    test_synthetic_composer()
    overall_time = time.time() - overall_start
    print(f"\n=== OVERALL TEST TIME: {overall_time:.4f}s ===")
