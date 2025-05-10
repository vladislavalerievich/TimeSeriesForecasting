import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)


@pytest.fixture
def generator():
    """Fixture for MultivariateTimeSeriesGenerator with default parameters."""
    return MultivariateTimeSeriesGenerator(
        global_seed=42,
        distribution_type="uniform",
        history_length=(64, 128),
        target_length=(32, 64),
        max_target_channels=5,
        num_channels=(1, 10),
        max_kernels=(1, 5),
        dirichlet_min=(0.1, 1.0),
        dirichlet_max=(1.0, 5.0),
        scale=(0.5, 2.0),
        weibull_shape=(1.0, 5.0),
        weibull_scale=(1, 3),
        periodicities=["s", "m", "h", "D", "W"],
    )


@pytest.fixture
def temp_dir():
    """Fixture for a temporary directory."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def test_parse_param_value_fixed(generator):
    """Test parsing fixed parameter values."""
    assert generator._parse_param_value(100, is_int=True) == 100
    assert generator._parse_param_value(0.5, is_int=False) == 0.5


def test_parse_param_value_range_uniform(generator):
    """Test parsing range with uniform distribution."""
    np.random.seed(42)
    value = generator._parse_param_value((50, 100), is_int=True)
    assert isinstance(value, int)
    assert 50 <= value <= 100


def test_parse_param_value_range_log_uniform(generator):
    """Test parsing range with log-uniform distribution."""
    generator.distribution_type = "log_uniform"
    np.random.seed(42)
    value = generator._parse_param_value((10, 1000), is_int=True)
    assert isinstance(value, int)
    assert 10 <= value <= 1000


def test_invalid_distribution_type(generator):
    """Test handling of invalid distribution type."""
    generator.distribution_type = "invalid"
    with pytest.raises(ValueError, match="Unknown distribution type: invalid"):
        generator._sample_from_range(1, 10)


def test_generate_batch_shapes(generator):
    """Test generate_batch for correct shapes and types."""
    batch_size, history_length, target_length, num_channels = 4, 64, 32, 3
    max_target_channels = 2

    # Generate batch
    result = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        max_target_channels=max_target_channels,
        seed=42,
    )

    # Assertions
    assert isinstance(result, BatchTimeSeriesContainer)
    assert result.history_values.shape == (batch_size, history_length, num_channels)
    assert result.target_values.shape[0] == batch_size
    assert result.target_values.shape[1] == target_length
    assert 1 <= result.target_values.shape[2] <= max_target_channels
    assert result.target_channels_indices.shape == (
        batch_size,
        result.target_values.shape[2],
    )
    assert result.history_time_features.shape == (batch_size, history_length, 1)
    assert result.target_time_features.shape == (batch_size, target_length, 1)
    assert torch.is_tensor(result.history_values)
    assert torch.is_tensor(result.target_values)
    assert torch.is_tensor(result.target_channels_indices)
    assert torch.is_tensor(result.history_time_features)
    assert torch.is_tensor(result.target_time_features)


def test_generate_batch_seed_reproducibility(generator):
    """Test generate_batch reproducibility with same seed."""
    batch_size, history_length, target_length, num_channels = 2, 32, 16, 2

    # Generate two batches with the same seed
    result1 = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        seed=42,
    )
    result2 = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        seed=42,
    )

    # Assertions
    assert torch.equal(result1.history_values, result2.history_values)
    assert torch.equal(result1.target_values, result2.target_values)
    assert torch.equal(result1.target_channels_indices, result2.target_channels_indices)


def test_format_to_container(generator):
    """Test format_to_container for correct splitting and tensor conversion."""
    batch_size, history_length, target_length, num_channels = 2, 32, 16, 3
    total_length = history_length + target_length
    max_target_channels = 2

    # Create mock data
    values = np.random.randn(batch_size, total_length, num_channels)
    timestamps = np.array(
        [
            [
                np.datetime64("2020-01-01") + np.timedelta64(i, "s")
                for i in range(total_length)
            ]
            for _ in range(batch_size)
        ]
    )

    # Format data
    np.random.seed(42)
    result = generator.format_to_container(
        values=values,
        timestamps=timestamps,
        history_length=history_length,
        target_length=target_length,
        batch_size=batch_size,
        num_channels=num_channels,
        max_target_channels=max_target_channels,
    )

    # Assertions
    assert result.history_values.shape == (batch_size, history_length, num_channels)
    assert result.target_values.shape[0] == batch_size
    assert result.target_values.shape[1] == target_length
    assert 1 <= result.target_values.shape[2] <= max_target_channels
    assert result.target_channels_indices.shape == (
        batch_size,
        result.target_values.shape[2],
    )
    assert result.history_time_features.shape == (batch_size, history_length, 1)
    assert result.target_time_features.shape == (batch_size, target_length, 1)
    assert torch.all(
        result.history_values
        == torch.tensor(values[:, :history_length, :], dtype=torch.float32)
    )
    assert torch.all(result.history_time_features >= 0)


def test_generate_dataset_yields_correctly(generator):
    """Test generate_dataset yields correct number of batches."""
    num_batches, batch_size = 3, 2

    # Use smaller dimensions for faster testing
    generator.history_length = 32
    generator.target_length = 16
    generator.num_channels = 2

    # Collect batches
    batches = list(
        generator.generate_dataset(
            num_batches=num_batches, batch_size=batch_size, num_cpus=1
        )
    )

    # Assertions
    assert len(batches) == num_batches
    for batch, batch_idx in batches:
        assert isinstance(batch, BatchTimeSeriesContainer)
        assert batch.history_values.shape[0] == batch_size
        assert batch_idx in range(num_batches)


def test_generate_dataset_parallel(generator):
    """Test generate_dataset parallel processing."""
    num_batches, batch_size = 4, 2

    # Use smaller dimensions for faster testing
    generator.history_length = 32
    generator.target_length = 16
    generator.num_channels = 2

    # Generate with multiple CPUs (use 2 for testing)
    batches = list(
        generator.generate_dataset(
            num_batches=num_batches, batch_size=batch_size, num_cpus=2
        )
    )

    # Assertions
    assert len(batches) == num_batches
    indices = [batch_idx for _, batch_idx in batches]
    # Since parallel execution may not guarantee order, just check all indices are present
    assert sorted(indices) == list(range(num_batches))


def test_save_dataset_individual_files(temp_dir):
    """Test save_dataset with individual batch files."""
    # Use a generator with small fixed parameters for faster testing
    generator = MultivariateTimeSeriesGenerator(
        global_seed=42,
        distribution_type="uniform",
        history_length=32,
        target_length=16,
        max_target_channels=2,
        num_channels=3,  # Fixed small value
        max_kernels=2,  # Fixed small value
        dirichlet_min=0.5,
        dirichlet_max=1.5,
        scale=1.0,
        weibull_shape=2.0,
        weibull_scale=1,
        periodicities=["s"],
    )

    # Save dataset with minimal batches
    generator.save_dataset(
        output_dir=temp_dir,
        num_batches=2,  # Minimal number
        batch_size=2,  # Minimal size
        save_as_single_file=False,
        num_cpus=1,
    )

    # Assertions
    assert os.path.exists(os.path.join(temp_dir, "batch_000.pt"))
    assert os.path.exists(os.path.join(temp_dir, "batch_001.pt"))
    batch = torch.load(os.path.join(temp_dir, "batch_000.pt"))
    assert isinstance(batch, BatchTimeSeriesContainer)

    # (batch_size, history_length, num_channels)
    assert batch.history_values.shape == (2, 32, 3)

    # (batch_size, target_length, max_target_channels)
    assert batch.target_values.shape == (2, 16, 2)

    # (batch_size, max_target_channels)
    assert batch.target_channels_indices.shape == (2, 2)

    # (batch_size, history_length, 1)
    assert batch.history_time_features.shape == (2, 32, 1)

    # (batch_size, target_length, 1)
    assert batch.target_time_features.shape == (2, 16, 1)


def test_save_dataset_single_file(generator, temp_dir):
    """Test save_dataset with single file."""
    # Use smaller parameters for faster testing
    generator.history_length = 32
    generator.target_length = 16
    generator.num_channels = 2

    num_batches, batch_size = 2, 2  # Minimal values for testing

    # Save dataset
    generator.save_dataset(
        output_dir=temp_dir,
        num_batches=num_batches,
        batch_size=batch_size,
        save_as_single_file=True,
        num_cpus=1,
        chunk_size=1,  # Use small chunk size for testing
    )

    # Assertions
    dataset_path = os.path.join(temp_dir, "dataset.pt")
    assert os.path.exists(dataset_path)
    dataset = torch.load(dataset_path)
    assert len(dataset) == num_batches
    assert all(isinstance(batch, BatchTimeSeriesContainer) for batch in dataset)


def test_edge_case_small_num_channels():
    """Test generate_batch with num_channels < max_target_channels."""
    num_channels = 1
    max_target_channels = 5

    # Expect a ValueError due to invalid configuration
    with pytest.raises(
        ValueError,
        match="max_target_channels \(.*\) cannot exceed the maximum value of num_channels \(.*\)",
    ):
        MultivariateTimeSeriesGenerator(
            global_seed=42,
            num_channels=num_channels,  # Fixed to 1
            max_target_channels=max_target_channels,
            history_length=32,
            target_length=16,
            max_kernels=1,
            dirichlet_min=0.5,
            dirichlet_max=1.5,
            scale=1.0,
            weibull_shape=2.0,
            weibull_scale=1,
            periodicities=["s"],
        )


def test_invalid_periodicity(generator):
    """Test generate_batch with invalid periodicity."""
    with pytest.raises(ValueError, match="Periodicity must be one of"):
        generator.generate_batch(
            batch_size=2,
            history_length=32,
            target_length=16,
            num_channels=2,
            periodicity="invalid",
        )


def test_max_target_channels_validation():
    """Test validation of max_target_channels parameter."""
    # Case 1: Fixed num_channels less than max_target_channels
    num_channels, max_target_channels = 3, 5
    error_msg_case1 = f"max_target_channels \\({max_target_channels}\\) cannot exceed the maximum value of num_channels \\({num_channels}\\)"
    with pytest.raises(
        ValueError,
        match=error_msg_case1,
    ):
        MultivariateTimeSeriesGenerator(
            num_channels=num_channels,
            max_target_channels=max_target_channels,
        )

    # Case 2: Range max num_channels less than max_target_channels
    error_msg_case2 = f"max_target_channels \\({max_target_channels}\\) cannot exceed the maximum value of num_channels \\(3\\)"
    with pytest.raises(
        ValueError,
        match=error_msg_case2,
    ):
        MultivariateTimeSeriesGenerator(
            num_channels=(1, 3),
            max_target_channels=5,
        )

    # Valid cases should not raise exceptions
    MultivariateTimeSeriesGenerator(num_channels=5, max_target_channels=5)
    MultivariateTimeSeriesGenerator(num_channels=(1, 10), max_target_channels=5)


def test_invalid_num_channels_range():
    """Test validation of num_channels with invalid range."""
    with pytest.raises(
        ValueError,
        match="For parameter 'num_channels', the minimum value \(10\) cannot exceed the maximum value \(5\)",
    ):
        MultivariateTimeSeriesGenerator(
            num_channels=(10, 5),  # Invalid range: min > max
            max_target_channels=3,
        )


def test_parse_param_value_equal_min_max(generator):
    """Test parsing when min and max values are equal."""
    value = generator._parse_param_value((100, 100), is_int=True)
    assert value == 100
    value = generator._parse_param_value((0.5, 0.5), is_int=False)
    assert value == 0.5


def test_sample_from_range_zero_range(generator):
    """Test sampling from a range with zero width."""
    value = generator._sample_from_range(10, 10, is_int=True)
    assert value == 10
    value = generator._sample_from_range(0.5, 0.5, is_int=False)
    assert value == 0.5


def test_generate_batch_minimal_values(generator):
    """Test generate_batch with minimal parameter values."""
    batch_size, history_length, target_length, num_channels = 1, 1, 1, 1
    max_target_channels = 1

    result = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        max_target_channels=max_target_channels,
        seed=42,
    )

    assert isinstance(result, BatchTimeSeriesContainer)
    assert result.history_values.shape == (batch_size, history_length, num_channels)
    assert result.target_values.shape == (batch_size, target_length, 1)
    assert result.target_channels_indices.shape == (batch_size, 1)
    assert result.history_time_features.shape == (batch_size, history_length, 1)
    assert result.target_time_features.shape == (batch_size, target_length, 1)


def test_generate_batch_max_target_channels_equals_num_channels(generator):
    """Test generate_batch when max_target_channels equals num_channels."""
    batch_size, history_length, target_length, num_channels = 2, 32, 16, 3
    max_target_channels = 3

    result = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        max_target_channels=max_target_channels,
        seed=42,
    )

    assert isinstance(result, BatchTimeSeriesContainer)
    assert result.history_values.shape == (batch_size, history_length, num_channels)
    assert result.target_values.shape[0] == batch_size
    assert result.target_values.shape[1] == target_length
    assert 1 <= result.target_values.shape[2] <= max_target_channels
    assert result.target_channels_indices.shape == (
        batch_size,
        result.target_values.shape[2],
    )


def test_generate_dataset_zero_batches(generator):
    """Test generate_dataset with zero batches."""
    num_batches, batch_size = 0, 2
    batches = list(
        generator.generate_dataset(
            num_batches=num_batches, batch_size=batch_size, num_cpus=1
        )
    )
    assert len(batches) == 0


def test_generate_dataset_single_cpu(generator):
    """Test generate_dataset with exactly one CPU."""
    num_batches, batch_size = 2, 2
    generator.history_length = 32
    generator.target_length = 16
    generator.num_channels = 2

    batches = list(
        generator.generate_dataset(
            num_batches=num_batches, batch_size=batch_size, num_cpus=1
        )
    )

    assert len(batches) == num_batches
    for batch, batch_idx in batches:
        assert isinstance(batch, BatchTimeSeriesContainer)
        assert batch.history_values.shape[0] == batch_size
        assert batch_idx in range(num_batches)


def test_save_dataset_chunk_size_equals_num_batches(temp_dir):
    """Test save_dataset with chunk_size equal to num_batches."""
    generator = MultivariateTimeSeriesGenerator(
        global_seed=42,
        distribution_type="uniform",
        history_length=32,
        target_length=16,
        max_target_channels=2,
        num_channels=3,
        max_kernels=2,
        dirichlet_min=0.5,
        dirichlet_max=1.5,
        scale=1.0,
        weibull_shape=2.0,
        weibull_scale=1,
        periodicities=["s"],
    )

    num_batches, batch_size = 2, 2
    generator.save_dataset(
        output_dir=temp_dir,
        num_batches=num_batches,
        batch_size=batch_size,
        save_as_single_file=True,
        num_cpus=1,
        chunk_size=num_batches,
    )

    dataset_path = os.path.join(temp_dir, "dataset.pt")
    assert os.path.exists(dataset_path)
    dataset = torch.load(dataset_path)
    assert len(dataset) == num_batches
    assert all(isinstance(batch, BatchTimeSeriesContainer) for batch in dataset)


def test_format_to_container_identical_timestamps(generator):
    """Test format_to_container with identical timestamps."""
    batch_size, history_length, target_length, num_channels = 2, 32, 16, 3
    total_length = history_length + target_length
    max_target_channels = 2

    values = np.random.randn(batch_size, total_length, num_channels)
    timestamps = np.array([[np.datetime64("2020-01-01")] * total_length] * batch_size)

    np.random.seed(42)
    result = generator.format_to_container(
        values=values,
        timestamps=timestamps,
        history_length=history_length,
        target_length=target_length,
        batch_size=batch_size,
        num_channels=num_channels,
        max_target_channels=max_target_channels,
    )

    assert result.history_time_features.shape == (batch_size, history_length, 1)
    assert result.target_time_features.shape == (batch_size, target_length, 1)
    assert torch.all(result.history_time_features == 0)
    assert torch.all(result.target_time_features == 0)


def test_initialization_invalid_range():
    """Test initialization with invalid parameter range."""
    with pytest.raises(
        ValueError, match="minimum value .* cannot exceed the maximum value .*"
    ):
        MultivariateTimeSeriesGenerator(
            global_seed=42,
            history_length=(100, 50),  # Invalid: min > max
            target_length=16,
            num_channels=3,
            max_target_channels=2,
        )


def test_dirichlet_min_max_swap(generator):
    """Test generate_dataset handling of dirichlet_min > dirichlet_max."""
    generator.dirichlet_min = (2.0, 3.0)
    generator.dirichlet_max = (0.5, 1.5)
    num_batches, batch_size = 2, 2

    batches = list(
        generator.generate_dataset(
            num_batches=num_batches, batch_size=batch_size, num_cpus=1
        )
    )

    assert len(batches) == num_batches
    for batch, batch_idx in batches:
        assert isinstance(batch, BatchTimeSeriesContainer)


def test_generate_batch_extreme_weibull_parameters(generator):
    """Test generate_batch with extreme Weibull parameters."""
    result = generator.generate_batch(
        batch_size=2,
        history_length=32,
        target_length=16,
        num_channels=2,
        weibull_shape=0.1,  # Very small shape
        weibull_scale=10,  # Large scale
        seed=42,
    )

    assert isinstance(result, BatchTimeSeriesContainer)
    assert result.history_values.shape == (2, 32, 2)
    assert result.target_values.shape[0] == 2
    assert result.target_values.shape[1] == 16
