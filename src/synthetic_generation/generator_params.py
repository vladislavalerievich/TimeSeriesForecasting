from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class GeneratorParams:
    global_seed: int = 42
    distribution_type: str = "uniform"
    history_length: Union[int, Tuple[int, int]] = (64, 256)
    target_length: Union[int, Tuple[int, int]] = (32, 256)
    num_channels: Union[int, Tuple[int, int]] = (1, 256)

    # Add additional generator-specific parameters as needed, e.g.:
    # max_kernels: Union[int, Tuple[int, int]] = (1, 5)
    # dirichlet_min: Union[float, Tuple[float, float]] = (0.1, 1.0)
    # dirichlet_max: Union[float, Tuple[float, float]] = (1.0, 5.0)
    # scale: Union[float, Tuple[float, float]] = (0.5, 2.0)
    # weibull_shape: Union[float, Tuple[float, float]] = (1.0, 5.0)
    # weibull_scale: Union[int, Tuple[int, int]] = (1, 3)

    def update(self, **kwargs):
        """Update parameters from keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class LMCGeneratorParams(GeneratorParams):
    max_kernels: int = 5
    dirichlet_min: float = 0.1
    dirichlet_max: float = 2.0
    scale: float = 1.0
    weibull_shape: float = 2.0
    weibull_scale: int = 1


@dataclass
class KernelGeneratorParams(GeneratorParams):
    max_kernels: int = 5
