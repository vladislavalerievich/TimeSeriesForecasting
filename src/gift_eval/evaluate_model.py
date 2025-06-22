import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
)
from gluonts.model import evaluate_model
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import get_seasonality

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.gift_eval.data import Dataset as GiftEvalDataset
from src.models.models import MultiStepModel
from src.synthetic_generation.common.constants import Frequency
from src.utils.utils import device

# Set up environment
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["GIFT_EVAL"] = "data/gift_eval"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# Configs
DATASET_PROPERTIES_PATH = "src/gift_eval/dataset_properties.json"
MODEL_PATH = "/home/moroshav/TimeSeriesForecasting/models/Old_GatedDeltaNet_rtx_2080_allow_neg_eigval_False_batch_size_64_num_epochs_150_initial_lr1e-05_learning_rate_1e-07_.pth"
TRAIN_YAML = "configs/train.yaml"
MAX_CONTEXT_LENGTH = 1024


# SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

# MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

SHORT_DATASETS = "us_births/D us_births/M us_births/W ett1/W ett2/W saugeenday/D saugeenday/M saugeenday/W"

MED_LONG_DATASETS = "bizitobs_l2c/H"

ALL_DATASETS = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
TERMS = ["short", "medium", "long"]

# Pretty names mapping (following GIFT eval standard)
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Metrics definition (consistent with TabPFN-TS and TiRex)
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(forecast_type=0.5),
    MASE(forecast_type=0.5),
    MAPE(forecast_type=0.5),
    SMAPE(forecast_type=0.5),
    MSIS(),
    RMSE(forecast_type=0.5),
    RMSE(forecast_type="mean"),
    NRMSE(forecast_type=0.5),
    ND(forecast_type=0.5),
]

# Load dataset properties
with open(DATASET_PROPERTIES_PATH, "r") as f:
    DATASET_PROPERTIES_MAP = json.load(f)

# Load model config
with open(TRAIN_YAML, "r") as f:
    config = yaml.safe_load(f)


# Set up logging filter to reduce noise
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)

# Set up logger
logger = logging.getLogger(__name__)


class MultiStepModelWrapper:
    """
    Wrapper class following standard GIFT eval interface patterns like TabPFN-TS and TiRex
    """

    def __init__(
        self,
        model,
        device,
        max_context_length=1024,
    ):
        self.model = model
        self.device = device
        self.max_context_length = max_context_length
        self.prediction_length = None
        self.ds_freq = None

    def set_prediction_len(self, prediction_length):
        """Set prediction length for the current dataset"""
        self.prediction_length = prediction_length

    def set_ds_freq(self, freq):
        """Set dataset frequency for the current dataset"""
        self.ds_freq = freq

    @property
    def model_id(self):
        """Model identifier for results"""
        return "MultiStepModel"

    def get_frequency_enum(self, freq_str):
        """Map frequency string to Frequency enum"""
        try:
            offset = pd.tseries.frequencies.to_offset(freq_str)
            standardized_freq = offset.name
        except Exception:
            # Fallback for direct frequency string
            standardized_freq = freq_str

        # Handle special cases and mappings
        freq_map = {
            # Annual frequencies
            "A-DEC": Frequency.A,
            "YE-DEC": Frequency.A,
            "A": Frequency.A,
            "Y": Frequency.A,
            # Quarterly frequencies
            "QS-DEC": Frequency.Q,
            "Q-DEC": Frequency.Q,
            "Q": Frequency.Q,
            # Monthly frequencies
            "MS": Frequency.ME,
            "M": Frequency.ME,
            "ME": Frequency.ME,
            # Weekly frequencies
            "W-MON": Frequency.W,
            "W-TUE": Frequency.W,
            "W-WED": Frequency.W,
            "W-THU": Frequency.W,
            "W-FRI": Frequency.W,
            "W-SAT": Frequency.W,
            "W-SUN": Frequency.W,
            "W": Frequency.W,
            # Daily frequencies
            "D": Frequency.D,
            # Hourly frequencies
            "H": Frequency.H,
            "h": Frequency.H,
            # Second frequencies
            "S": Frequency.S,
            "s": Frequency.S,
            # Minute frequencies
            "min": Frequency.T1,
        }

        # Handle minute frequencies (both old and new formats)
        if standardized_freq.endswith("T") or standardized_freq.endswith("min"):
            if standardized_freq.endswith("min"):
                # New format: "15min"
                if standardized_freq == "min":
                    return Frequency.T1
                minutes = int(standardized_freq[:-3])  # Remove "min"
            else:
                # Old format: "15T"
                minutes = int(standardized_freq[:-1])  # Remove "T"

            if minutes == 5:
                return Frequency.T5
            elif minutes == 10:
                return Frequency.T10
            elif minutes == 15:
                return Frequency.T15
            else:
                # Default to T1 for other minute frequencies
                return Frequency.T1
        elif standardized_freq in freq_map:
            return freq_map[standardized_freq]
        else:
            raise NotImplementedError(
                f"Frequency '{standardized_freq}' is not supported."
            )

    def predict(self, dataset) -> List[SampleForecast]:
        """Generate forecasts for the given dataset"""
        assert self.prediction_length is not None, "Prediction length must be set"
        forecasts = []

        for data_entry in dataset:
            target = data_entry["target"]
            start = data_entry["start"]
            item_id = data_entry.get("item_id", "ts")

            # if np.isnan(target).any():
            #     logger.warning(
            #         f"Target contains NaNs for {item_id} in dataset {getattr(dataset, 'name', 'unknown')}"
            #     )

            assert isinstance(start, pd.Period), (
                f"Expected pd.Period, got {type(start)}"
            )
            freq = start.freqstr

            history = np.asarray(target, dtype=np.float32)

            if history.ndim == 1:
                history = history.reshape(1, -1)

            num_dims, seq_len = history.shape

            if seq_len > self.max_context_length:
                history = history[:, -self.max_context_length :]
                seq_len = self.max_context_length

            history_values = torch.from_numpy(history.T).unsqueeze(0).to(self.device)

            start_time = np.array([start.start_time], dtype="datetime64")
            frequency = self.get_frequency_enum(freq)

            for dim_idx in range(num_dims):
                target_values = torch.zeros(
                    (1, self.prediction_length), dtype=torch.float32
                ).to(self.device)

                target_index = torch.tensor([dim_idx], dtype=torch.long).to(self.device)

                batch = BatchTimeSeriesContainer(
                    history_values=history_values,
                    target_values=target_values,
                    target_index=target_index,
                    start=start_time,
                    frequency=frequency,
                )

                with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                    with torch.no_grad():
                        output = self.model(batch, training=False)
                        predictions = (
                            self.model.scaler.inverse_transform(
                                output["result"],
                                output["scale_params"],
                                output["target_index"],
                            )
                            .cpu()
                            .numpy()
                        )
                if np.isnan(predictions).any():
                    logger.warning(
                        f"Predictions contain NaNs for {item_id} in dataset {getattr(dataset, 'name', 'unknown')}"
                    )
                # Reshape to (1, prediction_length) for SampleForecast
                if predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)

            forecast_start = start + seq_len

            forecast = SampleForecast(
                samples=predictions,
                start_date=forecast_start,
                item_id=item_id,  # Use the item_id as provided by the dataset
            )
            forecasts.append(forecast)
        return forecasts


def load_model():
    """Load the MultiStepModel from checkpoint"""
    model = MultiStepModel(
        base_model_config=config["BaseModelConfig"],
        encoder_config=config["EncoderConfig"],
        scaler=config["scaler"],
        **config["MultiStepModel"],
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def gift_eval_dataset_iter():
    """Iterator over all GIFT eval datasets and terms"""
    for ds_name in ALL_DATASETS:
        ds_key = ds_name.split("/")[0]
        for term in TERMS:
            # Skip medium/long terms for datasets not in MED_LONG_DATASETS
            if term in ["medium", "long"] and ds_name not in MED_LONG_DATASETS.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = PRETTY_NAMES.get(ds_key, ds_key)
                ds_freq = DATASET_PROPERTIES_MAP[ds_key]["frequency"]

            yield {
                "ds_name": ds_name,
                "ds_key": ds_key,
                "ds_freq": ds_freq,
                "term": term,
            }


def evaluate_dataset(predictor, ds_name, ds_key, ds_freq, term):
    """Evaluate a single dataset configuration"""
    logger.info(f"Processing dataset: {ds_name} ({term})")
    ds_config = f"{ds_key}/{ds_freq}/{term}"

    # The target needs to be univariate
    dataset = GiftEvalDataset(name=ds_name, term=term, to_univariate=False)

    # Set predictor parameters using standard interface
    predictor.set_prediction_len(dataset.prediction_length)
    predictor.set_ds_freq(ds_freq)

    # Get seasonality
    season_length = get_seasonality(dataset.freq)

    logger.info(f"Dataset size: {len(dataset.test_data)}")
    logger.info(f"Dataset freq: {dataset.freq}")
    logger.info(f"Dataset season_length: {season_length}")
    logger.info(f"Dataset prediction length: {dataset.prediction_length}")
    logger.info(f"Dataset target dim: {dataset.target_dim}")

    # Evaluate using GluonTS evaluate_model
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=METRICS,
        axis=None,
        batch_size=128,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    result = {
        "dataset": ds_config,
        "model": predictor.model_id,
        "eval_metrics/MSE[mean]": res["MSE[mean]"].iloc[0],
        "eval_metrics/MSE[0.5]": res["MSE[0.5]"].iloc[0],
        "eval_metrics/MAE[0.5]": res["MAE[0.5]"].iloc[0],
        "eval_metrics/MASE[0.5]": res["MASE[0.5]"].iloc[0],
        "eval_metrics/MAPE[0.5]": res["MAPE[0.5]"].iloc[0],
        "eval_metrics/sMAPE[0.5]": res["sMAPE[0.5]"].iloc[0],
        "eval_metrics/MSIS": res["MSIS"].iloc[0],
        "eval_metrics/RMSE[0.5]": res["RMSE[0.5]"].iloc[0],
        "eval_metrics/RMSE[mean]": res["RMSE[mean]"].iloc[0],
        "eval_metrics/NRMSE[0.5]": res["NRMSE[0.5]"].iloc[0],
        "eval_metrics/ND[0.5]": res["ND[0.5]"].iloc[0],
        "domain": DATASET_PROPERTIES_MAP[ds_key]["domain"],
        "num_variates": DATASET_PROPERTIES_MAP[ds_key]["num_variates"],
    }
    return result


def evaluate_on_gift_eval():
    """Main evaluation function"""
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Load model
    model = load_model()
    predictor = MultiStepModelWrapper(
        model, device, max_context_length=MAX_CONTEXT_LENGTH
    )

    results = []

    # Iterate through all datasets and terms
    for task in gift_eval_dataset_iter():
        try:
            task_result = evaluate_dataset(predictor, **task)
            results.append(task_result)
            logger.info(f"Completed evaluation for {task['ds_name']} ({task['term']})")
            print(task_result)
        except Exception as e:
            logger.error(
                f"Error evaluating {task['ds_name']} ({task['term']}): {str(e)}"
            )
            continue

    # Save results to CSV
    output_columns = [
        "dataset",
        "model",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[0.5]",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[0.5]",
        "eval_metrics/ND[0.5]",
        "domain",
        "num_variates",
    ]

    df = pd.DataFrame(results, columns=output_columns)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "gift_eval_results_multistepmodel.csv"
    df.to_csv(filename, index=False)

    logger.info(f"Results saved to {filename}")
    logger.info(f"Evaluated {len(results)} dataset configurations")

    return df


if __name__ == "__main__":
    evaluate_on_gift_eval()
