from abc import ABC, abstractmethod
from typing import Dict, Optional


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series data generators.
    All concrete generators must implement the generate_time_series method and define the length attribute.
    """

    def __init__(self, length: int):
        """
        Initialize the generator with the desired length of the time series.
        """
        self.length = length

    @abstractmethod
    def generate_time_series(self, random_seed: Optional[int], *args, **kwargs) -> Dict:
        """Generate a single synthetic time series."""
        pass
