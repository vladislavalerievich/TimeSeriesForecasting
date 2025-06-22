import logging
import os

import pytest

from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.data_loaders import (
    SyntheticTrainDataLoader,
    SyntheticValidationDataLoader,
)
from src.synthetic_generation.dataset_composer import (
    DatasetComposer,
    OnTheFlyDatasetGenerator,
)
from src.synthetic_generation.generator_params import GeneratorParams
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorParams,
    KernelGeneratorWrapper,
)
from src.synthetic_generation.lmc_synth.lmc_generator_wrapper import (
    LMCGeneratorParams,
    LMCGeneratorWrapper,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Fixtures ---
@pytest.fixture(scope="module")
def lmc_params():
    return LMCGeneratorParams(
        global_seed=123,
        history_length=8,
        target_length=4,
        num_channels=2,
        max_kernels=2,
        dirichlet_min=0.1,
        dirichlet_max=0.5,
        scale=1.0,
        weibull_shape=2.0,
        weibull_scale=1,
    )


@pytest.fixture(scope="module")
def kernel_params():
    return KernelGeneratorParams(
        global_seed=123,
        history_length=8,
        target_length=4,
        num_channels=2,
        max_kernels=2,
    )


@pytest.fixture(scope="module")
def lmc_wrapper(lmc_params):
    return LMCGeneratorWrapper(lmc_params)


@pytest.fixture(scope="module")
def kernel_wrapper(kernel_params):
    return KernelGeneratorWrapper(kernel_params)


# --- Tests for GeneratorParams ---
def test_generator_params_update():
    params = GeneratorParams()
    params.update(history_length=99, num_channels=7)
    assert params.history_length == 99
    assert params.num_channels == 7


# --- Tests for GeneratorWrapper (abstract) ---
def test_generator_wrapper_sample_parameters(lmc_wrapper):
    params = lmc_wrapper.sample_parameters()
    assert set(params.keys()) >= {
        "history_length",
        "target_length",
        "num_channels",
        "periodicity",
    }
    assert isinstance(params["history_length"], int)
    assert isinstance(params["target_length"], int)
    assert isinstance(params["num_channels"], int)
    assert params["history_length"] > 0
    assert params["target_length"] > 0
    assert params["num_channels"] > 0


# --- Tests for LMCGeneratorWrapper and KernelGeneratorWrapper ---
def test_lmc_kernel_generate_batch(lmc_wrapper, kernel_wrapper):
    for wrapper in [lmc_wrapper, kernel_wrapper]:
        batch = wrapper.generate_batch(batch_size=2)
        assert hasattr(batch, "history_values")
        assert hasattr(batch, "target_values")
        assert batch.history_values.shape == (2, 8, 2)
        assert batch.target_values.shape == (2, 4)
        assert batch.target_index.shape == (2,)
        assert batch.history_time_features.shape[0] == 2
        assert batch.target_time_features.shape[0] == 2


# --- Tests for DatasetComposer ---
def test_dataset_composer_batch_composition(lmc_wrapper, kernel_wrapper):
    composer = DatasetComposer({lmc_wrapper: 0.5, kernel_wrapper: 0.5}, global_seed=42)
    batch, name = composer.generate_batch(batch_size=2)
    assert hasattr(batch, "history_values")
    assert name in ("LMCGeneratorWrapper", "KernelGeneratorWrapper")
    # Test correct batch size
    assert batch.history_values.shape[0] == 2


# --- File-based tests for DatasetComposer and Validation Loader ---
def test_save_and_load_dataset(tmp_path, lmc_wrapper, kernel_wrapper):
    composer = DatasetComposer({lmc_wrapper: 0.5, kernel_wrapper: 0.5}, global_seed=42)
    out_dir = tmp_path / "synthetic_test_data"
    composer.save_dataset(output_dir=str(out_dir), num_batches=3, batch_size=2)
    loader = SyntheticValidationDataLoader(data_path=str(out_dir))
    batches = list(loader)
    assert len(batches) == 3
    for batch in batches:
        assert batch.history_values.shape == (2, 8, 2)
        assert batch.target_values.shape == (2, 4)


# --- On-the-fly generation and train loader ---
def test_on_the_fly_generation_and_loader(lmc_wrapper, kernel_wrapper):
    composer = DatasetComposer({lmc_wrapper: 0.5, kernel_wrapper: 0.5}, global_seed=42)
    on_the_fly = OnTheFlyDatasetGenerator(
        composer, batch_size=2, buffer_size=2, global_seed=42
    )
    loader = SyntheticTrainDataLoader(generator=on_the_fly, num_batches_per_epoch=3)
    batches = [batch for batch in loader]
    assert len(batches) == 3
    for batch in batches:
        assert batch.history_values.shape == (2, 8, 2)
        assert batch.target_values.shape == (2, 4)


# --- Test that proportions are respected in DatasetComposer ---
def test_dataset_composer_proportions(lmc_wrapper, kernel_wrapper):
    composer = DatasetComposer({lmc_wrapper: 1.0, kernel_wrapper: 0.0}, global_seed=42)
    for _ in range(3):
        batch, name = composer.generate_batch(batch_size=2)
        assert name == "LMCGeneratorWrapper"
    composer = DatasetComposer({lmc_wrapper: 0.0, kernel_wrapper: 1.0}, global_seed=42)
    for _ in range(3):
        batch, name = composer.generate_batch(batch_size=2)
        assert name == "KernelGeneratorWrapper"


# --- Edge case: test empty dataset directory ---
def test_empty_validation_loader(tmp_path):
    out_dir = tmp_path / "empty_dir"
    os.makedirs(out_dir, exist_ok=True)
    loader = SyntheticValidationDataLoader(data_path=str(out_dir))
    assert len(loader) == 0


def test_parameter_sampling_logic():
    # Test integer sampling
    params = {"history_length": 10, "future_length": 5, "num_channels": 1}
    gen_params = GeneratorParams(**params)
    wrapper = GeneratorWrapper(gen_params)
    sampled = wrapper._sample_parameters()
    assert sampled["history_length"] == 10
    assert sampled["future_length"] == 5
    assert sampled["num_channels"] == 1

    # Test tuple sampling
    params = {
        "history_length": (10, 20),
        "future_length": (1, 5),
        "num_channels": (1, 3),
    }
    gen_params = GeneratorParams(**params)
    wrapper = GeneratorWrapper(gen_params)
    sampled = wrapper._sample_parameters()
    assert 10 <= sampled["history_length"] <= 20
    assert 1 <= sampled["future_length"] <= 5
    assert 1 <= sampled["num_channels"] <= 3

    # Test list sampling
    params = {
        "history_length": [10, 20, 30],
        "future_length": [1, 2, 3],
        "num_channels": [1, 7],
    }
    gen_params = GeneratorParams(**params)
    wrapper = GeneratorWrapper(gen_params)
    sampled = wrapper._sample_parameters()
    assert sampled["history_length"] in [10, 20, 30]
    assert sampled["future_length"] in [1, 2, 3]
    assert sampled["num_channels"] in [1, 7]
