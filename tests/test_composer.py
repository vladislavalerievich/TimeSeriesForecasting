import hashlib
import logging
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from src.synthetic_generation.dataset_composer import DefaultSyntheticComposer

# Configure logging to show timing warnings
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def hash_time_series(values: torch.Tensor) -> str:
    """
    Create a hash of time series values for duplicate detection.

    Args:
        values: Tensor of shape [batch_size, seq_len, num_channels]

    Returns:
        String hash of the time series values
    """
    # Convert to numpy and create hash
    np_values = values.detach().cpu().numpy()
    # Round to avoid floating point precision issues
    rounded_values = np.round(np_values, decimals=8)
    return hashlib.md5(rounded_values.tobytes()).hexdigest()


def check_for_duplicates(all_batches_data, generator_counts):
    """
    Check for duplicate time series within and across batches.

    Args:
        all_batches_data: List of (batch_data, generator_name, batch_idx) tuples
        generator_counts: Counter of generator usage

    Returns:
        Dictionary with duplicate statistics
    """
    print("\n--- Checking for Duplicates ---")

    # Store hashes of all time series with their metadata
    all_time_series_hashes = []
    hash_to_metadata = defaultdict(list)

    total_time_series = 0

    for batch_data, generator_name, batch_idx in all_batches_data:
        batch_size = batch_data.history_values.shape[0]

        for series_idx in range(batch_size):
            # Extract individual time series (history + future)
            history_series = batch_data.history_values[
                series_idx : series_idx + 1
            ]  # Keep batch dim
            future_series = batch_data.future_values[
                series_idx : series_idx + 1
            ]  # Keep batch dim

            # Concatenate history and future for full series comparison
            full_series = torch.cat([history_series, future_series], dim=1)

            # Create hash
            series_hash = hash_time_series(full_series)
            all_time_series_hashes.append(series_hash)

            # Store metadata for this hash
            metadata = {
                "batch_idx": batch_idx,
                "series_idx": series_idx,
                "generator": generator_name,
                "shape": tuple(full_series.shape),
            }
            hash_to_metadata[series_hash].append(metadata)
            total_time_series += 1

    # Find duplicates
    duplicates = {h: meta for h, meta in hash_to_metadata.items() if len(meta) > 1}

    # Statistics
    unique_hashes = len(set(all_time_series_hashes))
    duplicate_count = total_time_series - unique_hashes

    print(f"Total time series generated: {total_time_series}")
    print(f"Unique time series: {unique_hashes}")
    print(f"Duplicate time series: {duplicate_count}")
    print(f"Duplicate rate: {duplicate_count / total_time_series * 100:.2f}%")

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} sets of duplicate time series:")

        for i, (series_hash, occurrences) in enumerate(duplicates.items()):
            print(f"\n  Duplicate Set {i + 1} (Hash: {series_hash[:8]}...):")
            print(f"    Occurs {len(occurrences)} times:")

            for occurrence in occurrences:
                print(
                    f"      - Batch {occurrence['batch_idx']}, Series {occurrence['series_idx']}, "
                    f"Generator: {occurrence['generator']}, Shape: {occurrence['shape']}"
                )
    else:
        print("✅ No duplicate time series found!")

    # Check for within-batch duplicates specifically
    within_batch_duplicates = []
    for series_hash, occurrences in duplicates.items():
        batch_groups = defaultdict(list)
        for occ in occurrences:
            batch_groups[occ["batch_idx"]].append(occ)

        for batch_idx, batch_occurrences in batch_groups.items():
            if len(batch_occurrences) > 1:
                within_batch_duplicates.append((batch_idx, batch_occurrences))

    if within_batch_duplicates:
        print(
            f"\n🚨 Found {len(within_batch_duplicates)} batches with internal duplicates:"
        )
        for batch_idx, occurrences in within_batch_duplicates:
            print(f"    Batch {batch_idx}: {len(occurrences)} duplicate series")

    return {
        "total_series": total_time_series,
        "unique_series": unique_hashes,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_count / total_time_series * 100,
        "duplicate_sets": len(duplicates),
        "within_batch_duplicates": len(within_batch_duplicates),
    }


def test_synthetic_composer():
    """
    Tests the DefaultSyntheticComposer by generating batches and inspecting their properties.

    This test verifies that the hierarchical composition works as expected by:
    1. Printing the configured generator proportions for each forecast range.
    2. Generating 100 batches and tracking their source and shapes.
    3. Printing a summary of the actual generated batches, including counts and
       the unique (history, future, channels) dimensions for each generator.
    4. Checking for duplicate time series across and within batches.
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
    batch_size = 1
    generator_counts = Counter()
    # A dictionary where each value is a set of unique shape tuples
    generator_shapes = defaultdict(set)
    batch_times = []

    # Store all batch data for duplicate checking
    all_batches_data = []

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

        # Store batch data for duplicate checking
        all_batches_data.append((batch, generator_full_name, i + 1))

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

    # --- 3. Check for Duplicates ---
    duplicate_stats = check_for_duplicates(all_batches_data, generator_counts)

    # --- 4. Process and Print Generation Results ---
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

    # --- 5. Final Summary ---
    print("\n--- Final Test Summary ---")
    print(f"✅ Generated {num_batches} batches successfully")

    # Count unique shape combinations across all generators
    all_unique_shapes = set()
    for shapes_set in generator_shapes.values():
        all_unique_shapes.update(shapes_set)
    print(
        f"✅ Parameter diversity: {len(all_unique_shapes)} unique shape combinations across all generators"
    )

    if duplicate_stats["duplicate_count"] == 0:
        print("✅ No duplicate time series detected")
    else:
        print(
            f"⚠️  {duplicate_stats['duplicate_count']} duplicate time series found ({duplicate_stats['duplicate_rate']:.2f}%)"
        )

    if duplicate_stats["within_batch_duplicates"] > 0:
        print(
            f"🚨 {duplicate_stats['within_batch_duplicates']} batches contain internal duplicates"
        )

    return duplicate_stats


if __name__ == "__main__":
    overall_start = time.time()
    test_synthetic_composer()
    overall_time = time.time() - overall_start
    print(f"\n=== OVERALL TEST TIME: {overall_time:.4f}s ===")
