import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
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
MAX_CONTEXT_LENGTH = 1024  # Can be set via argparse if desired

# Load dataset properties
with open(DATASET_PROPERTIES_PATH, "r") as f:
    dataset_properties_map = json.load(f)

# Load model config
with open(TRAIN_YAML, "r") as f:
    config = yaml.safe_load(f)


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

# Metrics
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

# Example: list of datasets/terms to evaluate
SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
ALL_DATASETS = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
TERMS = ["short", "medium", "long"]


def get_frequency_enum(freq_str):
    # Map freq string to Frequency enum
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


def evaluate_on_gift_eval():
    results = []
    for ds_name in ALL_DATASETS:
        for term in TERMS:
            # Only evaluate medium/long for datasets in MED_LONG_DATASETS
            if term in ["medium", "long"] and ds_name not in MED_LONG_DATASETS.split():
                continue
            print(f"Evaluating {ds_name} ({term})")

            dataset = GiftEvalDataset(name=ds_name, term=term, to_univariate=False)
            test_data = dataset.test_data
            prediction_length = dataset.prediction_length

            # freq = dataset.freq
            # season_length = get_seasonality(freq)
            predictions_list = []
            targets_list = []
            for entry in tqdm(test_data, desc=f"{ds_name}-{term}"):
                if isinstance(entry, tuple):
                    entry = entry[0]
                target = entry["target"]
                start = entry["start"]
                entry_freq = entry["freq"]
                if target.ndim == 1:
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
                            output = model(batch, prediction_length)
                            mean_targets, std_targets = output["scale_params"]
                            inv_scaled_output = (
                                output["result"] * std_targets
                            ) + mean_targets
                            predictions_list.append(inv_scaled_output.cpu().numpy())
                            targets_list.append(batch.target_values.cpu().numpy())
                else:
                    num_channels = target.shape[0]
                    full_history = target[:, :-prediction_length]
                    if full_history.shape[1] > MAX_CONTEXT_LENGTH:
                        full_history = full_history[:, -MAX_CONTEXT_LENGTH:]
                    history_tensor = torch.tensor(
                        full_history.T, dtype=torch.float32
                    ).unsqueeze(0)
                    for idx in range(num_channels):
                        target_values = target[idx, -prediction_length:]
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
                                output = model(batch, prediction_length)
                                mean_targets, std_targets = output["scale_params"]
                                inv_scaled_output = (
                                    output["result"] * std_targets
                                ) + mean_targets
                                predictions_list.append(inv_scaled_output.cpu().numpy())
                                targets_list.append(batch.target_values.cpu().numpy())
            # After all windows, collect metrics
            predictions_arr = np.concatenate(predictions_list, axis=0)
            targets_arr = np.concatenate(targets_list, axis=0)
            metrics_result = {
                type(m).__name__: m(predictions_arr, targets_arr) for m in METRICS
            }
            metrics_result.update(
                {"dataset": ds_name, "term": term, "model": "GatedDeltaNet"}
            )
            results.append(metrics_result)
    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv("gift_eval_results.csv", index=False)
    print("Results saved to gift_eval_results.csv")


if __name__ == "__main__":
    evaluate_on_gift_eval()
