from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.constants import BASE_END, BASE_START, FREQUENCY_MAPPING
from src.synthetic_generation.forecast_pfn_prior.series_config import (
    ComponentNoise,
    ComponentScale,
    SeriesConfig,
)
from src.synthetic_generation.forecast_pfn_prior.utils import (
    get_transition_coefficients,
    sample_scale,
    weibull_noise,
)
from src.synthetic_generation.generator_params import ForecastPFNGeneratorParams


class ForecastPFNGenerator(AbstractTimeSeriesGenerator):
    def __init__(self, params: ForecastPFNGeneratorParams, length: int = 1024):
        self.params = params
        self.length = length
        self.frequency = params.frequency

    def _generate_series(
        self, start: pd.Timestamp = None, random_seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        if random_seed is not None:
            np.random.seed(random_seed)

        freq_key, subfreq, timescale = FREQUENCY_MAPPING.get(
            self.frequency, ("D", "", 1)
        )
        freq = f"{subfreq}{freq_key}" if subfreq else freq_key

        # Seasonal component weights based on frequency
        a, m, w, h, minute = 0.0, 0.0, 0.0, 0.0, 0.0
        if freq_key == "min":
            minute = np.random.uniform(0.0, 1.0)
            h = np.random.uniform(0.0, 0.2)
        elif freq_key == "H":
            minute = np.random.uniform(0.0, 0.1)
            h = np.random.uniform(0.0, 1.0)
            w = np.random.uniform(0.0, 0.4)
        elif freq_key == "D":
            w = np.random.uniform(0.0, 1.0)
            m = np.random.uniform(0.0, 0.4)
            a = np.random.uniform(0.0, 0.2)
        elif freq_key == "W":
            m = np.random.uniform(0.0, 0.3)
            a = np.random.uniform(0.0, 0.8)
        elif freq_key == "MS":
            w = np.random.uniform(0.0, 0.1)
            a = np.random.uniform(0.0, 1.0)
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

        if start is None:
            start = pd.Timestamp.fromordinal(
                int((BASE_START - BASE_END) * beta.rvs(5, 1) + BASE_START)
            )

        scale_config = ComponentScale(
            base=1.0,
            linear=np.random.normal(0, 0.01),
            exp=min(1.01, np.random.normal(1, 0.005 / timescale))
            if self.params.trend_exp
            else 1.0,
            a=a,
            m=m,
            w=w,
            minute=minute,
            h=h,
        )

        offset_config = ComponentScale(
            base=0,
            linear=np.random.uniform(-0.1, 0.5),
            exp=np.random.uniform(-0.1, 0.5),
            a=np.random.uniform(0.0, 1.0),
            m=np.random.uniform(0.0, 1.0),
            w=np.random.uniform(0.0, 1.0),
        )

        noise_config = ComponentNoise(
            k=np.random.uniform(1, 5),
            median=1,
            scale=sample_scale(
                low_ratio=self.params.scale_noise[0],
                moderate_ratio=self.params.scale_noise[1],
            ),
        )

        cfg = SeriesConfig(scale_config, offset_config, noise_config)
        options = {
            "trend_exp": self.params.trend_exp,
            "scale_noise": self.params.scale_noise,
            "harmonic_scale_ratio": self.params.harmonic_scale_ratio,
            "harmonic_rate": self.params.harmonic_rate,
            "period_factor": self.params.period_factor,
            "seasonal_only": self.params.seasonal_only,
            "trend_additional": self.params.trend_additional,
        }

        # Generate first series
        series1 = self._make_series(
            cfg, to_offset(freq), start, options, self.params.random_walk
        )

        # Generate second series for transition if enabled
        transition = np.random.rand() < self.params.transition_ratio
        if transition:
            cfg2 = SeriesConfig(
                ComponentScale(
                    base=1.0,
                    linear=np.random.normal(0, 0.01),
                    exp=min(1.01, np.random.normal(1, 0.005 / timescale))
                    if self.params.trend_exp
                    else 1.0,
                    a=a,
                    m=m,
                    w=w,
                    minute=minute,
                    h=h,
                ),
                ComponentScale(
                    base=0,
                    linear=np.random.uniform(-0.1, 0.5),
                    exp=np.random.uniform(-0.1, 0.5),
                    a=np.random.uniform(0.0, 1.0),
                    m=np.random.uniform(0.0, 1.0),
                    w=np.random.uniform(0.0, 1.0),
                ),
                ComponentNoise(
                    k=np.random.uniform(1, 5),
                    median=1,
                    scale=sample_scale(
                        low_ratio=self.params.scale_noise[0],
                        moderate_ratio=self.params.scale_noise[1],
                    ),
                ),
            )
            series2 = self._make_series(
                cfg2, to_offset(freq), start, options, self.params.random_walk
            )
            coeff = get_transition_coefficients(self.length)
            values = coeff * series1["values"] + (1 - coeff) * series2["values"]
        else:
            values = series1["values"]

        return {"start": np.datetime64(start), "values": values}

    def _make_series(
        self,
        series: SeriesConfig,
        freq: pd.DateOffset,
        start: pd.Timestamp,
        options: dict,
        random_walk: bool,
    ) -> Dict:
        start = freq.rollback(start)
        dates = pd.date_range(start=start, periods=self.length, freq=freq)
        scaled_noise_term = 0

        if random_walk:
            values = self._get_random_walk_series(len(dates))
        elif options["seasonal_only"]:
            values_seasonal = self._make_series_seasonal(series, dates, options)
            values = values_seasonal["seasonal"]
        else:
            values_trend = self._make_series_trend(series, dates)
            values_seasonal = self._make_series_seasonal(series, dates, options)
            values = (
                values_trend + values_seasonal["seasonal"]
                if options["trend_additional"]
                else values_trend * values_seasonal["seasonal"]
            )

            weibull_noise_term = weibull_noise(
                k=series.noise_config.k,
                median=series.noise_config.median,
                length=len(values),
            )
            noise_expected_val = series.noise_config.median
            scaled_noise_term = series.noise_config.scale * (
                weibull_noise_term - noise_expected_val
            )
            values = values * (1 + scaled_noise_term)

        return {
            "values": values,
            "noise": 1 + scaled_noise_term,
            "dates": dates,
            "seasonal": values_seasonal["seasonal"]
            if not random_walk
            else np.ones_like(values),
        }

    def _make_series_trend(
        self, series: SeriesConfig, dates: pd.DatetimeIndex
    ) -> np.ndarray:
        values = np.full_like(dates, series.scale.base, dtype=np.float32)
        days = (dates - dates[0]).days
        if series.scale.linear is not None:
            values += self._shift_axis(days, series.offset.linear) * series.scale.linear
        if series.scale.exp is not None:
            values *= np.power(
                series.scale.exp, self._shift_axis(days, series.offset.exp)
            )
        return values

    def _make_series_seasonal(
        self, series: SeriesConfig, dates: pd.DatetimeIndex, options: dict
    ) -> Dict:
        seasonal = 1
        harmonic_scale = np.random.rand() < options["harmonic_scale_ratio"]
        harmonic_rate = options["harmonic_rate"]
        period_factor = options["period_factor"]
        seasonal_components = {}
        if series.scale.minute is not None and series.scale.minute != 0:
            seasonal_components["minute"] = (
                1
                + series.scale.minute
                * self._get_freq_component(
                    dates.minute,
                    int(np.ceil(10 * harmonic_rate)),
                    60 * period_factor,
                    harmonic_scale,
                )
            )
            seasonal *= seasonal_components["minute"]
        if series.scale.h is not None and series.scale.h != 0:
            seasonal_components["h"] = 1 + series.scale.h * self._get_freq_component(
                dates.hour,
                int(np.ceil(10 * harmonic_rate)),
                24 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["h"]
        if series.scale.a is not None and series.scale.a != 0:
            seasonal_components["a"] = 1 + series.scale.a * self._get_freq_component(
                dates.month,
                int(np.ceil(6 * harmonic_rate)),
                12 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["a"]
        if series.scale.m is not None and series.scale.m != 0:
            seasonal_components["m"] = 1 + series.scale.m * self._get_freq_component(
                dates.day,
                int(np.ceil(10 * harmonic_rate)),
                30.5 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["m"]
        if series.scale.w is not None and series.scale.w != 0:
            seasonal_components["w"] = 1 + series.scale.w * self._get_freq_component(
                dates.dayofweek,
                int(np.ceil(4 * harmonic_rate)),
                7 * period_factor,
                harmonic_scale,
            )
            seasonal *= seasonal_components["w"]
        seasonal_components["seasonal"] = seasonal
        return seasonal_components

    def _get_freq_component(
        self,
        dates_feature: pd.Index,
        n_harmonics: int,
        n_total: float,
        harmonic_scale: bool = True,
    ) -> np.ndarray:
        harmonics = list(range(1, n_harmonics + 1))
        sin_coef = np.zeros(n_harmonics)
        cos_coef = np.zeros(n_harmonics)
        for idx, harmonic in enumerate(harmonics):
            h = 1 if not harmonic_scale else harmonic
            sin_coef[idx] = np.random.normal(scale=1 / h)
            cos_coef[idx] = np.random.normal(scale=1 / h)
        coef_sq_sum = np.sqrt(np.sum(np.square(sin_coef)) + np.sum(np.square(cos_coef)))
        sin_coef /= coef_sq_sum
        cos_coef /= coef_sq_sum
        return_val = 0
        for idx, harmonic in enumerate(harmonics):
            return_val += sin_coef[idx] * np.sin(
                2 * np.pi * harmonic * dates_feature / n_total
            )
            return_val += cos_coef[idx] * np.cos(
                2 * np.pi * harmonic * dates_feature / n_total
            )
        return return_val

    def _shift_axis(self, days: np.ndarray, shift: float) -> np.ndarray:
        if shift is None:
            return days
        return days - shift * days[-1]

    def _get_random_walk_series(self, length: int, movements=[-1, 1]) -> np.ndarray:
        random_walk = [np.random.choice(movements)]
        for _ in range(1, length):
            movement = np.random.choice(movements)
            random_walk.append(random_walk[-1] + movement)
        return np.array(random_walk)

    def generate_time_series(
        self, random_seed: Optional[int] = None, periodicity: Frequency = None
    ) -> Dict[str, np.ndarray]:
        if periodicity is not None:
            self.frequency = periodicity
        return self._generate_series(random_seed=random_seed)
