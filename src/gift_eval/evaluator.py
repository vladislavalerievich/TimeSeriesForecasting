import json
import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# --- MODIFICATION START: Import MeanWeightedSumQuantileLoss ---
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
# --- MODIFICATION END ---
from gluonts.model import evaluate_model
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import get_seasonality

import wandb
from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.frequency_utils import get_frequency_enum
from src.gift_eval.data import Dataset as GiftEvalDataset
from src.plotting.plot_multivariate_timeseries import plot_multivariate_timeseries

# Set up logger
logger = logging.getLogger(__name__)

# Constants
DATASET_PROPERTIES_PATH = "src/gift_eval/dataset_properties.json"
# --- MODIFICATION START: METRICS list is removed from globals ---
# It will be defined dynamically inside the evaluator class.
# --- MODIFICATION END ---
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


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

    def predict(self, dataset) -> List[SampleForecast]:
        """Generate forecasts for the given dataset"""
        assert self.prediction_length is not None, "Prediction length must be set"
        forecasts = []

        for data_entry in dataset:
            try:
                if isinstance(data_entry, tuple):
                    input_data = data_entry[0]
                else:
                    input_data = data_entry

                target = input_data["target"]
                start = input_data["start"]
                item_id = input_data.get("item_id", "ts")

                assert isinstance(start, pd.Period), (
                    f"Expected pd.Period, got {type(start)}"
                )
                freq = start.freqstr
                history = np.asarray(target, dtype=np.float32)
            except Exception as e:
                logger.error(
                    f"Error processing data_entry: {type(data_entry)}, error: {str(e)}"
                )
                raise

            if history.ndim == 1:
                history = history.reshape(1, -1)

            num_dims, seq_len = history.shape

            if seq_len > self.max_context_length:
                history = history[:, -self.max_context_length:]
                seq_len = self.max_context_length

            history_values = torch.from_numpy(history.T).unsqueeze(0).to(self.device)
            frequency = get_frequency_enum(freq)
            future_values = torch.zeros(
                (1, self.prediction_length, num_dims), dtype=torch.float32
            ).to(self.device)

            batch = BatchTimeSeriesContainer(
                history_values=history_values,
                future_values=future_values,
                start=start.to_timestamp().to_numpy(),
                frequency=frequency,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
                    output = self.model(batch, drop_enc_allow=False)
                    predictions = self.model.scaler.inverse_scale(
                        output["result"],
                        output["scale_statistics"],
                    )

            if torch.isnan(predictions).any():
                logger.warning(
                    f"NaN predictions with half precision for {item_id}. Retrying with full precision."
                )
                with torch.no_grad():
                    output = self.model(batch, drop_enc_allow=False)
                    predictions = self.model.scaler.inverse_scale(
                        output["result"],
                        output["scale_statistics"],
                    )

            predictions = predictions.cpu()

            if torch.isnan(predictions).any():
                nan_count = torch.isnan(predictions).sum()
                total_count = predictions.numel()
                logger.warning(
                    f"Predictions contain {nan_count}/{total_count} NaNs ({nan_count / total_count * 100:.1f}%) "
                    f"for {item_id} in dataset {getattr(dataset, 'name', 'unknown')}"
                )
                predictions = torch.nan_to_num(predictions, nan=0.0)

            predictions = predictions.squeeze(0)

            if self.model.loss_type == 'quantile':
                samples = predictions.permute(2, 0, 1).numpy()
            else:
                samples = predictions.unsqueeze(0).numpy()

            forecast_start = start + seq_len
            forecast = SampleForecast(
                samples=samples,
                start_date=forecast_start,
                item_id=item_id,
            )
            forecasts.append(forecast)
        return forecasts


class GiftEvaluator:
    def __init__(self, model, device, max_context_length: int):
        self.model = model
        self.device = device
        self.predictor = MultiStepModelWrapper(model, device, max_context_length)

        with open(DATASET_PROPERTIES_PATH, "r") as f:
            self.dataset_properties_map = json.load(f)

    def evaluate_datasets(
            self,
            datasets_to_eval: List[str],
            term: str,
            epoch: int = None,
            plot: bool = False,
    ):
        all_results = {}
        for ds_name in datasets_to_eval:
            ds_key, ds_freq = self._get_ds_key_freq(ds_name)
            task = {
                "ds_name": ds_name,
                "ds_key": ds_key,
                "ds_freq": ds_freq,
                "term": term,
            }
            try:
                metrics, forecasts, plot_samples = self._evaluate_single_dataset(task)
                all_results[f"{ds_name}_{term}"] = metrics

                if plot and epoch is not None and forecasts and plot_samples:
                    self._plot_and_log_samples(plot_samples, forecasts, task, epoch)

            except Exception as e:
                logger.error(
                    f"Error evaluating {task['ds_name']} ({task['term']}): {str(e)}"
                )
                continue
        return all_results

    def _get_ds_key_freq(self, ds_name):
        if "/" in ds_name:
            ds_key, ds_freq = ds_name.split("/")
        else:
            ds_key = ds_name
            ds_key_lower = ds_key.lower()
            ds_key_mapped = PRETTY_NAMES.get(ds_key_lower, ds_key_lower)
            if ds_key_mapped in self.dataset_properties_map:
                ds_freq = self.dataset_properties_map[ds_key_mapped]["frequency"]
            else:
                ds_freq = "D"

        ds_key = ds_key.lower()
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
        return ds_key, ds_freq

    def _evaluate_single_dataset(
            self, task: Dict
    ) -> Tuple[Dict, List[SampleForecast], List[Tuple[Dict, Dict]]]:
        ds_name, ds_freq, term = (
            task["ds_name"],
            task["ds_freq"],
            task["term"],
        )

        to_univariate = (
            False
            if GiftEvalDataset(name=ds_name, term=term, to_univariate=False).target_dim
               == 1
            else True
        )
        dataset = GiftEvalDataset(name=ds_name, term=term, to_univariate=to_univariate)
        self.predictor.set_prediction_len(dataset.prediction_length)
        self.predictor.set_ds_freq(ds_freq)

        test_data = dataset.test_data

        plot_samples_with_labels = list(test_data)[:1] if test_data else []
        plot_samples_inputs = (
            [sample[0] for sample in plot_samples_with_labels]
            if plot_samples_with_labels
            else []
        )
        forecasts = (
            self.predictor.predict(plot_samples_inputs) if plot_samples_inputs else []
        )

        season_length = get_seasonality(dataset.freq)

        # --- MODIFICATION START: Define metrics dynamically ---
        metrics_to_run = [
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

        if self.model.loss_type == 'quantile':
            metrics_to_run.append(
                MeanWeightedSumQuantileLoss(quantile_levels=self.model.quantiles)
            )
        # --- MODIFICATION END ---

        res = evaluate_model(
            self.predictor,
            test_data=test_data,
            metrics=metrics_to_run,
            axis=None,
            batch_size=128,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        # --- MODIFICATION START: Collect MeanWeightedSumQuantileLoss ---
        metrics = {
            "MSE[mean]": res["MSE[mean]"].iloc[0],
            "MSE[0.5]": res["MSE[0.5]"].iloc[0],
            "MAE[0.5]": res["MAE[0.5]"].iloc[0],
            "MASE[0.5]": res["MASE[0.5]"].iloc[0],
            "MAPE[0.5]": res["MAPE[0.5]"].iloc[0],
            "sMAPE[0.5]": res["sMAPE[0.5]"].iloc[0],
            "MSIS": res["MSIS"].iloc[0],
            "RMSE[0.5]": res["RMSE[0.5]"].iloc[0],
            "RMSE[mean]": res["RMSE[mean]"].iloc[0],
            "NRMSE[0.5]": res["NRMSE[0.5]"].iloc[0],
            "ND[0.5]": res["ND[0.5]"].iloc[0],
        }
        if self.model.loss_type == 'quantile':
            metrics["mean_weighted_sum_quantile_loss"] = res["mean_weighted_sum_quantile_loss"].iloc[0]
        # --- MODIFICATION END ---

        return metrics, forecasts, plot_samples_with_labels

    def _plot_and_log_samples(self, plot_samples, forecasts, task, epoch):
        """Generate and log sample plots for GIFT evaluation datasets."""
        ds_name, term = task["ds_name"], task["term"]
        max_context = self.predictor.max_context_length

        if plot_samples and forecasts:
            try:
                input_data, label_data = plot_samples[0]
                forecast = forecasts[0]

                history_values = np.asarray(input_data["target"], dtype=np.float32)
                future_values = np.asarray(label_data["target"], dtype=np.float32)

                if history_values.ndim == 1:
                    history_values = history_values.reshape(1, -1)
                if future_values.ndim == 1:
                    future_values = future_values.reshape(1, -1)

                if history_values.shape[1] > max_context:
                    history_values = history_values[:, -max_context:]

                predicted_values = forecast.samples

                start_timestamp = input_data["start"]
                frequency = task["ds_freq"]

                fig = plot_multivariate_timeseries(
                    history_values=history_values.T,
                    future_values=future_values.T,
                    predicted_values=predicted_values.T,
                    start=start_timestamp.to_timestamp()
                    if hasattr(start_timestamp, "to_timestamp")
                    else start_timestamp,
                    frequency=frequency,
                    title=f"GIFT-Eval: {ds_name} ({term}) - Epoch {epoch}",
                    show=False,
                )

                clean_dataset_name = ds_name.replace("/", "_").replace(" ", "_")
                plot_key = f"gift_eval_plots/{term}/{clean_dataset_name}"

                wandb.log({plot_key: wandb.Image(fig)})
                plt.close(fig)
                logger.debug(
                    f"Successfully logged plot for {ds_name} ({term}) - Epoch {epoch}"
                )

            except Exception as e:
                logger.warning(f"Failed to plot sample for {ds_name}/{term}: {str(e)}")