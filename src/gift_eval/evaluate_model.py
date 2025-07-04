import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
    MeanWeightedSumQuantileLoss,
)

from src.gift_eval.evaluator import GiftEvaluator
from src.models.unified_model import TimeSeriesModel
from src.utils.utils import device

# Set up environment
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["GIFT_EVAL"] = "data/gift_eval"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# Configs
DATASET_PROPERTIES_PATH = "src/gift_eval/dataset_properties.json"
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
    MeanWeightedSumQuantileLoss(
        quantiles=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ),
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


def load_model(model_path: str):
    """Load the MultiStepModel from checkpoint"""
    model = TimeSeriesModel(
        base_model_config=config["BaseModelConfig"],
        encoder_config=config["EncoderConfig"],
        scaler=config["scaler"],
        **config["TimeSeriesModel"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_on_gift_eval(model_path: str):
    """Main evaluation function"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load model
    model = load_model(model_path)
    evaluator = GiftEvaluator(model, device, max_context_length=MAX_CONTEXT_LENGTH)

    results = []

    # Iterate through all datasets and terms
    for term in TERMS:
        datasets_for_term = ALL_DATASETS
        if term in ["medium", "long"]:
            datasets_for_term = MED_LONG_DATASETS.split()

        term_results = evaluator.evaluate_datasets(
            datasets_to_eval=datasets_for_term, term=term, plot=False
        )

        for ds_key, metrics in term_results.items():
            # Robustly extract the dataset name from the key
            ds_name = ds_key[: -len(term) - 1]

            if "/" in ds_name:
                base_ds_name = ds_name.split("/")[0]
            else:
                base_ds_name = ds_name

            base_ds_name = PRETTY_NAMES.get(base_ds_name, base_ds_name)

            result_entry = {
                "dataset": f"{ds_name}/{term}",
                "model": "MultiStepModel",
                "eval_metrics/MSE[mean]": metrics["MSE[mean]"],
                "eval_metrics/MSE[0.5]": metrics["MSE[0.5]"],
                "eval_metrics/MAE[0.5]": metrics["MAE[0.5]"],
                "eval_metrics/MASE[0.5]": metrics["MASE[0.5]"],
                "eval_metrics/MAPE[0.5]": metrics["MAPE[0.5]"],
                "eval_metrics/sMAPE[0.5]": metrics["sMAPE[0.5]"],
                "eval_metrics/MSIS": metrics["MSIS"],
                "eval_metrics/RMSE[0.5]": metrics["RMSE[0.5]"],
                "eval_metrics/RMSE[mean]": metrics["RMSE[mean]"],
                "eval_metrics/NRMSE[0.5]": metrics["NRMSE[0.5]"],
                "eval_metrics/ND[0.5]": metrics["ND[0.5]"],
                "domain": DATASET_PROPERTIES_MAP[base_ds_name]["domain"],
                "num_variates": DATASET_PROPERTIES_MAP[base_ds_name]["num_variates"],
            }
            results.append(result_entry)

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
    parser = argparse.ArgumentParser(
        description="Evaluate a MultiStepModel on GIFT-Eval."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    args = parser.parse_args()

    evaluate_on_gift_eval(args.model_path)
