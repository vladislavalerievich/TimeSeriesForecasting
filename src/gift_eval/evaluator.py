import json
import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
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
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class MultiStepModelWrapper:
    """
    Wrapper class following standard GIFT eval interface patterns.
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
        self.prediction_length = prediction_length

    def set_ds_freq(self, freq):
        self.ds_freq = freq

    @property
    def model_id(self):
        return "MultiStepModel"

    def predict(self, dataset) -> List[SampleForecast]:
        assert self.prediction_length is not None, "Prediction length must be set"
        forecasts = []
        for data_entry in dataset:
            try:
                # This handles both dicts and tuples from different dataset loaders
                input_data = (
                    data_entry[0] if isinstance(data_entry, tuple) else data_entry
                )

                target = input_data["target"]
                start = input_data["start"]
                item_id = input_data.get("item_id", "ts")
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
                history = history[:, -self.max_context_length :]
                seq_len = self.max_context_length

            history_values = torch.from_numpy(history.T).unsqueeze(0).to(self.device)
            frequency = get_frequency_enum(freq)
            future_values = torch.zeros(
                (1, self.prediction_length, num_dims), dtype=torch.float32
            ).to(self.device)

            start_np = start.to_timestamp().to_numpy()

            batch = BatchTimeSeriesContainer(
                history_values=history_values,
                future_values=future_values,
                start=start_np,
                frequency=frequency,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
                    output = self.model(batch, drop_enc_allow=False)
                    predictions = self.model.scaler.inverse_scale(
                        output["result"], output["scale_statistics"]
                    )

            if torch.isnan(predictions).any():
                logger.warning(
                    f"NaN predictions with half precision for {item_id}. Retrying with full precision."
                )
                with torch.no_grad():
                    output = self.model(batch, drop_enc_allow=False)
                    predictions = self.model.scaler.inverse_scale(
                        output["result"], output["scale_statistics"]
                    )

            predictions = predictions.cpu()
            if torch.isnan(predictions).any():
                nan_count = torch.isnan(predictions).sum()
                total_count = predictions.numel()
                logger.warning(
                    f"Predictions contain {nan_count}/{total_count} NaNs for {item_id}"
                )
                predictions = torch.nan_to_num(predictions, nan=0.0)

            predictions = predictions.squeeze(0)

            # Handle quantile vs non-quantile predictions
            if self.model.loss_type == "quantile":
                # For quantile predictions, shape is [P, N, Q] -> samples should be [Q, P, N]
                samples = predictions.permute(2, 0, 1).numpy()
            else:
                # For non-quantile predictions, shape is [P, N] -> add sample dimension
                samples = predictions.unsqueeze(0).numpy()

            forecast_start = start + seq_len
            forecast = SampleForecast(
                samples=samples, start_date=forecast_start, item_id=item_id
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
                    f"Error evaluating {task['ds_name']} ({task['term']}): {str(e)}",
                    exc_info=False,
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
            ds_freq = self.dataset_properties_map.get(ds_key_mapped, {}).get(
                "frequency", "D"
            )
        ds_key = ds_key.lower()
        ds_key = PRETTY_NAMES.get(ds_key, ds_key)
        return ds_key, ds_freq

    def _evaluate_single_dataset(
        self, task: Dict
    ) -> Tuple[Dict, List[SampleForecast], List[Dict]]:
        ds_name, ds_freq, term = task["ds_name"], task["ds_freq"], task["term"]

        is_multivariate = (
            GiftEvalDataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
        )
        dataset = GiftEvalDataset(
            name=ds_name, term=term, to_univariate=is_multivariate
        )

        self.predictor.set_prediction_len(dataset.prediction_length)
        self.predictor.set_ds_freq(ds_freq)
        test_data = dataset.test_data

        plot_samples = list(test_data)[:1] if test_data else []
        forecasts = self.predictor.predict(plot_samples) if plot_samples else []

        season_length = get_seasonality(dataset.freq)
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

        # Add quantile-specific metrics if model uses quantile loss
        if self.model.loss_type == "quantile" and self.model.quantiles:
            metrics_to_run.append(
                MeanWeightedSumQuantileLoss(quantile_levels=self.model.quantiles)
            )

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

        # Add quantile loss metric if applicable
        if self.model.loss_type == "quantile" and self.model.quantiles:
            metrics["mean_weighted_sum_quantile_loss"] = res[
                "mean_weighted_sum_quantile_loss"
            ].iloc[0]

        return metrics, forecasts, plot_samples

    def _plot_and_log_samples(
        self,
        plot_samples: List,
        forecasts: List[SampleForecast],
        task: Dict,
        epoch: int,
    ):
        """Generate and log sample plots for GIFT evaluation datasets."""
        ds_name, term = task["ds_name"], task["term"]
        max_context = self.predictor.max_context_length

        if not (plot_samples and forecasts):
            return

        try:
            data_entry = plot_samples[0]
            forecast = forecasts[0]

            # Handle tuple or dict data entries robustly
            if isinstance(data_entry, tuple):
                input_dict, label_dict = data_entry
                history_values = np.asarray(input_dict["target"], dtype=np.float32)
                future_values = np.asarray(label_dict["target"], dtype=np.float32)
                start_period = input_dict["start"]
            else:  # It's a single dictionary
                prediction_length = self.predictor.prediction_length
                full_target = np.asarray(data_entry["target"], dtype=np.float32)
                history_values = full_target[:-prediction_length]
                future_values = full_target[-prediction_length:]
                start_period = data_entry["start"]

            if history_values.ndim == 1:
                history_values = history_values.reshape(1, -1)
            if future_values.ndim == 1:
                future_values = future_values.reshape(1, -1)
            if history_values.shape[1] > max_context:
                history_values = history_values[:, -max_context:]

            all_quantile_samples = forecast.samples
            median_prediction, lower_bound, upper_bound = None, None, None

            # Handle quantile vs non-quantile predictions for plotting
            if self.model.loss_type == "quantile" and self.model.quantiles:
                try:
                    q_list = self.model.quantiles
                    median_idx = q_list.index(0.5)
                    lower_idx = q_list.index(0.1) if 0.1 in q_list else None
                    upper_idx = q_list.index(0.9) if 0.9 in q_list else None

                    median_prediction = all_quantile_samples[median_idx].T

                    # Only use bounds if both 0.1 and 0.9 quantiles are available
                    if lower_idx is not None and upper_idx is not None:
                        lower_bound = all_quantile_samples[lower_idx].T
                        upper_bound = all_quantile_samples[upper_idx].T
                    else:
                        logger.warning(
                            f"0.1 and/or 0.9 quantiles not found for plotting {ds_name}. Plotting median only."
                        )

                except (ValueError, IndexError):
                    logger.warning(
                        f"Could not find required quantiles for plotting {ds_name}. Plotting median only."
                    )
                    median_prediction = np.median(all_quantile_samples, axis=0).T
            else:
                # For non-quantile models, use the single prediction
                median_prediction = all_quantile_samples[0].T

            start_timestamp = start_period.to_timestamp()
            frequency = task["ds_freq"]

            fig = plot_multivariate_timeseries(
                history_values=history_values,
                future_values=future_values,
                predicted_values=median_prediction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                start=start_timestamp,
                frequency=frequency,
                title=f"GIFT-Eval: {ds_name} ({term}) - Epoch {epoch}",
                show=False,
            )

            clean_dataset_name = ds_name.replace("/", "_").replace(" ", "_")
            plot_key = f"gift_eval_plots/{term}/{clean_dataset_name}"
            wandb.log({plot_key: wandb.Image(fig)})
            plt.close(fig)

        except Exception as e:
            logger.warning(
                f"Failed to plot sample for {ds_name}/{term}: {e}", exc_info=False
            )