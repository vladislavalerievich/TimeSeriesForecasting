import json
import os
import pprint
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from gluonts.ev.metrics import MetricCollection
from tqdm import tqdm

# Set up environment
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["GIFT_EVAL"] = "data/gift_eval"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

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
    MeanWeightedSumQuantileLoss,
)
from gluonts.time_feature import get_seasonality

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.gift_eval.data import Dataset as GiftEvalDataset
from src.models.models import MultiStepModel
from src.utils.utils import device

# Configs
DATASET_PROPERTIES_PATH = "src/gift_eval/dataset_properties.json"
MODEL_PATH = "models/GatedDeltaNet_rtx_2080_allow_neg_eigval_False_batch_size_64_num_epochs_150_initial_lr1e-05_learning_rate_1e-07_.pth"
TRAIN_YAML = "configs/training/train.yaml"
MAX_CONTEXT_LENGTH = 1024

# Load dataset properties
with open(DATASET_PROPERTIES_PATH, "r") as f:
    dataset_properties_map = json.load(f)

# Load model config
with open(TRAIN_YAML, "r") as f:
    config = yaml.safe_load(f)

# Updated METRIC_CONFIGS using gluonts.ev.metrics
METRIC_CONFIGS = {
    "MAE": (lambda: MAE(), "MAE[0.5]"),
    "MSE": (lambda: MSE(forecast_type="0.5"), "MSE[0.5]"),
    "MSE_MEAN": (lambda: MSE(forecast_type="mean"), "MSE[mean]"),
    "MASE": (lambda: MASE(), "MASE[0.5]"),
    "MAPE": (lambda: MAPE(), "MAPE[0.5]"),
    "SMAPE": (lambda: SMAPE(), "sMAPE[0.5]"),
    "MSIS": (lambda: MSIS(), "MSIS"),
    "RMSE": (lambda: RMSE(forecast_type="0.5"), "RMSE[0.5]"),
    "RMSE_MEAN": (lambda: RMSE(forecast_type="mean"), "RMSE[mean]"),
    "NRMSE": (lambda: NRMSE(forecast_type="0.5"), "NRMSE[0.5]"),
    "NRMSE_MEAN": (lambda: NRMSE(forecast_type="mean"), "NRMSE[mean]"),
    "ND": (lambda: ND(), "ND[0.5]"),
    "WQTL": (
        lambda: MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        "mean_weighted_sum_quantile_loss",
    ),
}


# Instantiate model (as in trainer.py)
def load_model():
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


model = load_model()


def get_frequency_enum(freq_str):
    """Map freq string to Frequency enum"""
    freq_map = {
        "M": Frequency.M,
        "W": Frequency.W,
        "D": Frequency.D,
        "h": Frequency.H,
        "S": Frequency.S,
        "s": Frequency.S,
        "5min": Frequency.T5,
        "10min": Frequency.T10,
        "15min": Frequency.T15,
    }
    return freq_map.get(freq_str, Frequency.D)


def create_forecast_data(targets, predictions, seasonality=None, ds_name=None):
    """
    Create forecast data structure expected by GluonTS metrics.

    Args:
        targets: Ground truth values, shape [num_samples, prediction_length]
        predictions: Model predictions, shape [num_samples, prediction_length]
        seasonality: Optional seasonality for MASE/MSIS computation

    Returns:
        List of forecast dictionaries compatible with GluonTS metrics
    """
    forecast_data = []

    for i in range(len(targets)):
        target = targets[i]
        pred = predictions[i]

        # Check for zero labels
        if np.any(target == 0):
            print(
                f"Warning: Zero values found in target at index {i} of {ds_name}, may cause issues with MAPE"
            )

        # Create forecast dictionary
        forecast_dict = {
            "label": target,
            "0.5": pred,  # Median forecast
            "mean": pred,  # Mean forecast (same as median for point forecasts)
        }

        # Add quantile forecasts (including MSIS-required quantiles)
        quantiles = [0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]
        for q in quantiles:
            forecast_dict[str(q)] = pred

        # Add seasonal error if provided (for MASE/MSIS)
        if seasonality is not None:
            forecast_dict["seasonal_error"] = (
                seasonality[i] if hasattr(seasonality, "__len__") else seasonality
            )

        forecast_data.append(forecast_dict)

    return forecast_data


def compute_seasonal_error(history, freq_str, prediction_length):
    """
    Compute seasonal error for MASE/MSIS metrics.

    Args:
        history: Historical time series data
        freq_str: Frequency string
        prediction_length: Length of prediction horizon

    Returns:
        Seasonal error value
    """
    try:
        # Get seasonality period
        seasonality = get_seasonality(freq_str)
        if seasonality is None or seasonality <= 1:
            # Use naive seasonal error (mean absolute difference)
            if len(history) > 1:
                return np.mean(np.abs(np.diff(history)))
            else:
                return 1.0

        # Compute seasonal naive error
        if len(history) >= seasonality:
            seasonal_diffs = np.abs(history[seasonality:] - history[:-seasonality])
            return np.mean(seasonal_diffs) if len(seasonal_diffs) > 0 else 1.0
        else:
            # Fallback to naive error
            if len(history) > 1:
                return np.mean(np.abs(np.diff(history)))
            else:
                return 1.0
    except Exception as e:
        print(f"Error computing seasonal error: {e}")
        # Fallback to simple error measure
        if len(history) > 1:
            return np.mean(np.abs(np.diff(history)))
        else:
            return 1.0


# Example: list of datasets/terms to evaluate
SHORT_DATASETS = "m4_weekly"
MED_LONG_DATASETS = "bizitobs_l2c/H"
ALL_DATASETS = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
TERMS = ["short", "medium", "long"]


def evaluate_on_gift_eval():
    results = []
    output_columns = [
        "dataset",
        "model",
        *(col for _, col in METRIC_CONFIGS.values()),
        "domain",
        "num_variates",
    ]

    for ds_name in ALL_DATASETS:
        for term in TERMS:
            # Only evaluate medium/long for datasets in MED_LONG_DATASETS
            if term in ["medium", "long"] and ds_name not in MED_LONG_DATASETS.split():
                continue

            print(f"Evaluating {ds_name} ({term})")

            dataset = GiftEvalDataset(name=ds_name, term=term, to_univariate=False)
            test_data = dataset.test_data
            prediction_length = dataset.prediction_length

            predictions_list = []
            targets_list = []
            histories_list = []

            for entry in tqdm(test_data, desc=f"{ds_name}-{term}"):
                if isinstance(entry, tuple):
                    entry = entry[0]

                target = entry["target"]
                start = entry["start"]
                entry_freq = entry["freq"].replace("H", "h")

                if target.ndim == 1:
                    # Univariate case
                    history = target[:-prediction_length]
                    target_values = target[-prediction_length:]

                    if len(history) > MAX_CONTEXT_LENGTH:
                        history = history[-MAX_CONTEXT_LENGTH:]

                    batch = BatchTimeSeriesContainer(
                        history_values=torch.tensor(history, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(-1),
                        target_values=torch.tensor(
                            target_values, dtype=torch.float32
                        ).unsqueeze(0),
                        target_index=torch.tensor([0]),
                        start=np.array([np.datetime64(start)]),
                        frequency=get_frequency_enum(entry_freq),
                    )
                    batch.to_device(device)

                    with torch.autocast(
                        device_type="cuda", dtype=torch.half, enabled=True
                    ):
                        with torch.no_grad():
                            output = model(batch, training=False)
                            inv_scaled_output = model.scaler.inverse_transform(
                                output["result"],
                                output["scale_params"],
                                output["target_index"],
                            )
                            predictions_list.append(inv_scaled_output.cpu().numpy())
                            targets_list.append(batch.target_values.cpu().numpy())
                            histories_list.append(history)

                else:
                    # Multivariate case
                    num_channels = target.shape[0]
                    full_history = target[:, :-prediction_length]

                    if full_history.shape[1] > MAX_CONTEXT_LENGTH:
                        full_history = full_history[:, -MAX_CONTEXT_LENGTH:]

                    history_tensor = torch.tensor(
                        full_history.T, dtype=torch.float32
                    ).unsqueeze(0)

                    for idx in range(num_channels):
                        target_values = target[idx, -prediction_length:]
                        history_values = full_history[idx, :]

                        batch = BatchTimeSeriesContainer(
                            history_values=history_tensor,
                            target_values=torch.tensor(
                                target_values, dtype=torch.float32
                            ).unsqueeze(0),
                            target_index=torch.tensor([idx]),
                            start=np.array([np.datetime64(start)]),
                            frequency=get_frequency_enum(entry_freq),
                        )
                        batch.to_device(device)

                        with torch.autocast(
                            device_type="cuda", dtype=torch.half, enabled=True
                        ):
                            with torch.no_grad():
                                output = model(
                                    batch, training=False, drop_enc_allow=False
                                )
                                inv_scaled_output = model.scaler.inverse_transform(
                                    output["result"],
                                    output["scale_params"],
                                    output["target_index"],
                                )
                                predictions_list.append(inv_scaled_output.cpu().numpy())
                                targets_list.append(batch.target_values.cpu().numpy())
                                histories_list.append(history_values)

            # Collect all predictions and targets
            predictions_arr = np.concatenate(predictions_list, axis=0)
            targets_arr = np.concatenate(targets_list, axis=0)

            if predictions_arr.shape[-1] == 1:
                predictions_arr = np.squeeze(predictions_arr, axis=-1)

            # Compute seasonal errors for MASE/MSIS
            seasonal_errors = []
            for hist in histories_list:
                seasonal_error = compute_seasonal_error(
                    hist, entry_freq, prediction_length
                )
                seasonal_errors.append(seasonal_error)
            seasonal_errors = np.array(seasonal_errors)

            # Create forecast data structure
            forecast_data = create_forecast_data(
                targets_arr, predictions_arr, seasonal_errors, ds_name
            )

            # Build result row
            row = {}
            row["dataset"] = f"{ds_name}/{dataset.freq}/{term}"
            row["model"] = "GatedDeltaNet"

            # Compute metrics using GluonTS metrics
            metrics = [
                metric_factory()(axis=None)
                for metric_factory, _ in METRIC_CONFIGS.values()
            ]
            metric_collection = MetricCollection(metrics)
            metric_collection.update_all(forecast_data)
            metric_values = metric_collection.get()

            for metric_key, (_, col_name) in METRIC_CONFIGS.items():
                try:
                    value = metric_values.get(col_name)
                    row[col_name] = float(value) if value is not None else None
                except Exception as e:
                    print(f"Metric {col_name} failed: {e}")
                    row[col_name] = None

            # Add domain and num_variates
            row["domain"] = dataset_properties_map.get(ds_name, {}).get("domain", "")
            row["num_variates"] = dataset_properties_map.get(ds_name, {}).get(
                "num_variates", getattr(dataset, "target_dim", 1)
            )

            results.append(row)

    # Save all results to CSV in the requested column order
    df = pd.DataFrame(results, columns=output_columns)
    print("Results:")
    pprint.pprint(df)
    filename = "outputs/gift_eval_results.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    evaluate_on_gift_eval()
