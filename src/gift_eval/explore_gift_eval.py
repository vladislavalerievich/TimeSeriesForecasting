import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.util import to_pandas

from src.gift_eval.data import Dataset

# Get the GIFT_EVAL path from environment variables
gift_eval_path = os.getenv("GIFT_EVAL", "data/gift_eval")

if gift_eval_path:
    # Convert to Path object for easier manipulation
    gift_eval_path = Path(gift_eval_path)

    # Get all subdirectories (dataset names) in the GIFT_EVAL path
    dataset_names = []
    unique_freqs = set()  # e.g. 'D', 'W', 'M', 'Y', 'H', 'T', 'S', 'ms', 'us', 'ns'

    for dataset_dir in gift_eval_path.iterdir():
        if dataset_dir.name.startswith("."):
            continue
        if dataset_dir.is_dir():
            freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if freq_dirs:
                for freq_dir in freq_dirs:
                    dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
                    unique_freqs.add(freq_dir.name)
            else:
                dataset_names.append(dataset_dir.name)

    print("Available datasets in GIFT_EVAL:")
    for name in sorted(dataset_names):
        print(f"- {name}")
    print("Unique frequencies from directory structure:", sorted(unique_freqs))
else:
    print(
        "GIFT_EVAL path not found in environment variables. Please check your .env file."
    )
    exit(1)


def is_m4_dataset(dataset_name):
    """Check if dataset is an M4 dataset (stored differently)"""
    return dataset_name.startswith("m4_")


def extract_freq_from_m4_name(dataset_name):
    """Extract frequency from M4 dataset name"""
    freq_map = {
        "m4_daily": "D",
        "m4_hourly": "H",
        "m4_monthly": "M",
        "m4_quarterly": "Q",
        "m4_weekly": "W",
        "m4_yearly": "Y",
    }
    return freq_map.get(dataset_name, None)


# --- Code for collecting unique frequencies and context/forecast lengths ---
unique_freqs_actual = set()
context_length_counter_val = Counter()
forecast_length_counter_val = Counter()
context_length_counter_test = Counter()
forecast_length_counter_test = Counter()

# Track results for each term
term_results = defaultdict(list)  # term -> list of (dataset_name, freq)
context_length_by_term_val = defaultdict(Counter)  # term -> Counter
forecast_length_by_term_val = defaultdict(Counter)  # term -> Counter
context_length_by_term_test = defaultdict(Counter)  # term -> Counter
forecast_length_by_term_test = defaultdict(Counter)  # term -> Counter

# Track number of series per dataset
series_counts = defaultdict(lambda: defaultdict(set))

# Track which datasets were successfully processed
processed_datasets = []
failed_datasets = []

print("\n" + "=" * 60)
print("PROCESSING DATASETS FOR FREQUENCY AND LENGTH ANALYSIS")
print("=" * 60)

for ds_name in sorted(dataset_names):
    print(f"Processing: {ds_name}")

    # Try all terms and collect results for each successful one
    successful_terms = []

    for term in ["short", "medium", "long"]:
        try:
            print(f"  Trying term: {term}")
            ds = Dataset(name=ds_name, term=term, to_univariate=False)

            series_counts[ds.target_dim][ds_name].add(ds.prediction_length)

            # Get frequency from actual data
            train_data_iter = ds.training_dataset
            train_data = next(iter(train_data_iter))
            actual_freq = str(train_data.get("freq", "unknown"))  # Convert to string
            unique_freqs_actual.add(actual_freq)

            # Track this successful combination
            term_results[term].append((ds_name, actual_freq))
            successful_terms.append(term)

            # Process validation data - get context length from validation dataset
            try:
                val_data_iter = ds.validation_dataset
                val_data = next(iter(val_data_iter))
                val_freq = str(val_data.get("freq", "unknown"))
                unique_freqs_actual.add(val_freq)

                # Handle multivariate data
                val_context_len = (
                    val_data["target"].shape[1]
                    if ds.target_dim > 1 and len(val_data["target"].shape) > 1
                    else len(val_data["target"])
                )
                context_length_counter_val[val_context_len] += 1
                context_length_by_term_val[term][val_context_len] += 1

                # Get actual forecast length from Dataset property
                forecast_len = ds.prediction_length
                forecast_length_counter_val[forecast_len] += 1
                forecast_length_by_term_val[term][forecast_len] += 1

            except Exception as e:
                print(f"    Warning: Could not process validation data: {e}")

            # Process test data
            try:
                test_data_iter = ds.test_data
                test_sample = next(iter(test_data_iter))

                # Handle different test data formats
                if isinstance(test_sample, tuple):
                    test_input, test_label = test_sample
                else:
                    # If not a tuple, try to get input and label separately
                    test_input_iter = ds.test_data.input
                    test_label_iter = ds.test_data.label
                    test_input = next(iter(test_input_iter))
                    test_label = next(iter(test_label_iter))

                # Collect frequencies
                test_input_freq = str(test_input.get("freq", "unknown"))
                test_label_freq = str(test_label.get("freq", "unknown"))
                unique_freqs_actual.add(test_input_freq)
                unique_freqs_actual.add(test_label_freq)

                # Count lengths for multivariate and univariate data
                context_len = (
                    test_input["target"].shape[1]
                    if ds.target_dim > 1 and len(test_input["target"].shape) > 1
                    else len(test_input["target"])
                )
                forecast_len = (
                    test_label["target"].shape[1]
                    if ds.target_dim > 1 and len(test_label["target"].shape) > 1
                    else len(test_label["target"])
                )

                # Log mismatch between test forecast length and prediction_length
                if forecast_len != ds.prediction_length:
                    print(
                        f"    Mismatch: Test forecast length ({forecast_len}) != "
                        f"prediction_length ({ds.prediction_length}) for {ds_name}, term={term}"
                    )

                context_length_counter_test[context_len] += 1
                forecast_length_counter_test[forecast_len] += 1
                context_length_by_term_test[term][context_len] += 1
                forecast_length_by_term_test[term][forecast_len] += 1

            except Exception as e:
                print(f"    Warning: Could not process test data: {e}")

            print(
                f"  ✓ Successfully processed with term '{term}', freq: {actual_freq}, pred_len: {ds.prediction_length}, num_variates: {ds.target_dim}"
            )

        except Exception as e:
            print(f"    Failed with term '{term}': {e}")
            continue

    if successful_terms:
        processed_datasets.extend(
            [(ds_name, term, actual_freq) for term in successful_terms]
        )
        print(f"  ✓ Successfully processed with terms: {successful_terms}")
    else:
        failed_datasets.append(ds_name)
        print(f"  ✗ Failed to process {ds_name} with any term")

    print()

# Print results
print("=" * 60)
print("ANALYSIS RESULTS")
print("=" * 60)

print("\nSuccessfully processed datasets by term:")
for term in sorted(term_results.keys()):
    print(f"\n{term.upper()} term ({len(term_results[term])} datasets):")
    for ds_name, freq in sorted(term_results[term]):
        print(f"  {ds_name} (freq: {freq})")

if failed_datasets:
    print(f"\nFailed to process {len(failed_datasets)} datasets:")
    for ds_name in failed_datasets:
        print(f"  {ds_name}")

print(f"\nUnique actual frequencies found in data: {sorted(unique_freqs_actual)}")


# Sort counters from highest to lowest
print("\nALL VALIDATION context length counts (sorted by count):")
for length, count in context_length_counter_val.most_common():
    print(f"  Context Length {length}: {count} datasets")

print("\nALL FORECAST length counts (from Dataset.prediction_length, sorted by count):")
for length, count in forecast_length_counter_val.most_common():
    print(f"  Forecast Length {length}: {count} datasets")

print("\nALL TEST context length counts (sorted by count):")
for length, count in context_length_counter_test.most_common():
    print(f"  Context Length {length}: {count} datasets")

print("\nALL TEST forecast length counts (from test labels, sorted by count):")
for length, count in forecast_length_counter_test.most_common():
    print(f"  Forecast Length {length}: {count} datasets")

# Show breakdown by term
for term in sorted(term_results.keys()):
    print(
        f"\n{term.upper()} TERM - Forecast length counts (Dataset.prediction_length):"
    )
    for length, count in forecast_length_by_term_val[term].most_common():
        print(f"  Forecast Length {length}: {count} datasets")

    print(f"\n{term.upper()} TERM - Test forecast length counts (from test labels):")
    for length, count in forecast_length_by_term_test[term].most_common():
        print(f"  Forecast Length {length}: {count} datasets")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Total unique datasets: {len(dataset_names)}")
print(
    f"Datasets that worked with at least one term: {len(set(ds for ds, _, _ in processed_datasets))}"
)
print(f"Total successful dataset-term combinations: {len(processed_datasets)}")
print(f"Failed datasets: {len(failed_datasets)}")
print(
    f"Success rate: {len(set(ds for ds, _, _ in processed_datasets)) / len(dataset_names) * 100:.1f}%"
)
print(f"Unique frequencies found: {len(unique_freqs_actual)}")

# Show term distribution
term_counts = Counter(term for _, term, _ in processed_datasets)
print("\nTerm distribution:")
for term, count in term_counts.most_common():
    print(f"  {term}: {count} datasets")

print(f"Unique validation context lengths: {len(context_length_counter_val)}")
print(f"Unique test context lengths: {len(context_length_counter_test)}")
print(f"Unique forecast lengths: {len(forecast_length_counter_test)}")


print("\n" + "=" * 60)
print("DATASET SIZES (from dataset_info.json)")
print("=" * 60)

dataset_sizes = []

for ds_name in sorted(dataset_names):
    dataset_info_path = gift_eval_path / ds_name / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, "r") as f:
            try:
                info = json.load(f)
                size = info.get("dataset_size")
                name_from_json = info.get("dataset_name", ds_name)
                if size is not None:
                    dataset_sizes.append((name_from_json, size))
                else:
                    print(f"  - Could not find 'dataset_size' in {dataset_info_path}")
            except json.JSONDecodeError:
                print(f"  - Error decoding JSON from {dataset_info_path}")
    else:
        print(f"  - dataset_info.json not found for {ds_name}")

# Sort datasets by size
dataset_sizes.sort(key=lambda x: x[1])

print("\nDatasets sorted by size (smallest to largest):")
for name, size in dataset_sizes:
    print(f"  - {name}: {size}")

print("\n" + "=" * 60)
print("NUMBER OF SERIES (target_dim) ANALYSIS")
print("=" * 60)

sorted_target_dims = sorted(series_counts.keys())

for dim in sorted_target_dims:
    type_of_series = "Univariate" if dim == 1 else "Multivariate"
    print(
        f"\n{type_of_series} datasets (target_dim = {dim}): {len(series_counts[dim])} datasets"
    )
    # Sort datasets by name
    sorted_datasets = sorted(series_counts[dim].items(), key=lambda item: item[0])
    for ds_name, horizons in sorted_datasets:
        sorted_horizons = sorted(list(horizons))
        print(f"  - {ds_name} (horizons: {sorted_horizons})")


ds_name = "bizitobs_l2c/5T"  # Name of the dataset
to_univariate = False  # Whether to convert the data to univariate
term = "medium"  # Term of the dataset

print("\n" + "=" * 60)
print(f"EXPLORING DATASET {ds_name}")
print("=" * 60)

dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
print("Dataset frequency: ", dataset.freq)
print("Prediction length: ", dataset.prediction_length)
print("Number of windows in the rolling evaluation: ", dataset.windows)
print("Number of series in the dataset: ", len(dataset.training_dataset))

train_data_iter = dataset.training_dataset  # Get the training data iterator

train_data = next(iter(train_data_iter))
print("Keys in the training data: ", train_data.keys())

if "past_feat_dynamic_real" in train_data:
    print(
        "Shape of past_feat_dynamic_real: ",
        train_data["past_feat_dynamic_real"].shape,
    )
    print(
        "Sample of past_feat_dynamic_real: ",
        train_data["past_feat_dynamic_real"][:, :5],
    )
else:
    print("past_feat_dynamic_real not found in this dataset.")

print("Item id: ", train_data["item_id"])
print("Start Date: ", train_data["start"])
print("Frequency: ", train_data["freq"])
print("Target shape: ", train_data["target"].shape)

if train_data["target"].ndim > 1:
    # Multivariate case: create a DataFrame
    train_df = pd.DataFrame(
        train_data["target"].T,
        index=pd.period_range(
            start=train_data["start"],
            periods=train_data["target"].shape[1],
            freq=train_data["freq"],
        ),
    )
    train_df.plot()
else:
    # Univariate case: use to_pandas
    train_series = to_pandas(train_data)
    train_series.plot()

plt.grid(which="both")
plt.legend(
    [f"train series dim {i}" for i in range(train_data["target"].shape[0])],
    loc="upper left",
)
plt.show()


val_data_iter = dataset.validation_dataset

val_data = next(iter(val_data_iter))
print("Keys in the validation data: ", val_data.keys())

print("Item id: ", val_data["item_id"])
print("Start Date: ", val_data["start"])
print("Frequency: ", val_data["freq"])

test_split_iter = dataset.test_data
test_data = next(iter(test_split_iter))

test_input_split_iter = dataset.test_data.input

input = next(iter(test_input_split_iter))
print("Keys in the test data: ", input.keys())

print("\n\nContext Item id: ", input["item_id"])
print("Context Start Date: ", input["start"])
print("Context Frequency: ", input["freq"])
print("Context Length: ", len(input["target"]))

test_label_split_iter = dataset.test_data.label
label = input = next(iter(test_label_split_iter))
print("\n\nForecast Item id: ", label["item_id"])
print("Forecast Start Date: ", label["start"])
print("Forecast Frequency: ", label["freq"])
print("Forecast Length: ", len(label["target"]))
