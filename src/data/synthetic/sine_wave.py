from datetime import date

import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta

from src.data.synthetic.constants import BASE_END, BASE_START, freq_dict
from src.models.models import MultiStepModel

config = {
    "enc_conv": True,
    "init_dil_conv": True,
    "enc_conv_kernel": 5,
    "init_conv_kernel": 5,
    "init_conv_max_dilation": 3,
    "global_residual": False,
    "in_proj_norm": False,
    "initial_gelu_flag": True,
    "linear_seq": 15,
    "norm": True,
    "norm_type": "layernorm",
    "num_encoder_layers": 2,
    "residual": False,
    "token_embed_len": 1024,
}


def generate_sine_waves(N, n_pi1=2, n_pi2=4, lin_lim=5, noise=False):
    """
    Generate a sine wave of length 2*pi with N evenly spaced points.

    Parameters:
    N (int): The number of points to generate.

    Returns:
    np.ndarray: The sine wave values.
    np.ndarray: The corresponding x values.
    """
    # Generate N evenly spaced points between 0 and 2*pi
    x_sin1 = np.linspace(0, 2 * n_pi1 * np.pi, N)
    x_sin2 = np.linspace(0, 2 * n_pi2 * np.pi, N)

    y_lin = np.linspace(0, lin_lim, N)
    # Compute the sine of these points
    y_sin1 = np.sin(x_sin1)
    y_sin2 = np.sin(x_sin2)

    y = np.stack([y_sin1, y_sin1 + y_lin, y_sin1 * y_sin2, y_sin1 * y_lin], axis=0)

    if noise:
        y += np.random.normal(0, 0.05, N)

    return y


def scale_data(output, scaler):
    if scaler == "custom_robust":
        output = (output["result"] * output["scale"][1].squeeze(-1)) + output["scale"][
            0
        ].squeeze(-1)
    elif scaler == "min_max":
        output = (
            output["result"]
            * (output["scale"][0].squeeze(-1) - output["scale"][1].squeeze(-1))
        ) + output["scale"][1].squeeze(-1)
    elif scaler == "identity":
        output = output["result"]
    return output


def multipoint_predict(
    model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device
):
    x = {}
    x["ts"] = batch_x_mark.to(device)
    x["history"] = batch_x.to(device)
    x["target_dates"] = batch_y_mark.to(device)
    x["task"] = torch.zeros(4, pred_len).int().to(device)
    output = model(x, pred_len)
    output = scale_data(output, scaler)
    output = output.detach().cpu()
    return output


model_string = "models.pth"
subday = True


def eval_sine_cl_error(model, periods, full_context_len, pred_len, noise, device):
    # Generate a sine wave
    periods = periods
    seq_len = full_context_len + pred_len
    x = generate_sine_waves(seq_len, periods[0], periods[1], noise=noise)

    # generate time stamps
    freq = freq_dict[np.random.choice(["daily", "weekly", "monthly"])]["freq"]
    start = pd.Timestamp(
        date.fromordinal(int((BASE_START - BASE_END) * beta.rvs(5, 1) + BASE_START))
    )
    ts = pd.date_range(start=start, periods=seq_len, freq=to_offset(freq))

    ts = np.stack(
        [
            ts.year.values,
            ts.month.values,
            ts.day.values,
            ts.day_of_week.values + 1,
            ts.day_of_year.values,
        ],
        axis=-1,
    )

    mses = {}

    for context_len in range(40 + pred_len, full_context_len):
        # Create a batch
        batch = {}
        batch["ts"] = torch.tensor(ts[-context_len:-pred_len]).unsqueeze(0)
        batch["history"] = torch.tensor(x[-context_len:-pred_len]).unsqueeze(0)

        batch["target_dates"] = torch.tensor(ts[full_context_len:]).unsqueeze(0)

        output = multipoint_predict(
            model,
            batch["history"],
            batch["ts"],
            batch["target_dates"],
            pred_len,
            "min_max",
            device,
        )

        mse = np.mean((output - x[-pred_len:]) ** 2)
        mae = np.abs(output - x[-pred_len:]).mean()
        mses[context_len] = {"mse": mse, "mae": mae}

    mse_df = pd.DataFrame.from_dict(mses, orient="index", columns=["mse", "mae"])
    return mse_df


def adapt_state_dict_keys(old_state_dict):
    new_state_dict = {}

    for key in old_state_dict.keys():
        if "linear_layer" in key:
            # Replace "linear_layer" with "stage_2_layer.0"
            new_key = key.replace("linear_layer", "stage_2_layer.0")

            # Add the updated key to the new state dict
            new_state_dict[new_key] = old_state_dict[key]
        else:
            # Keep other keys unchanged
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepModel(scaler="min_max", sub_day=subday, **config).to(device)
    new_state_dict = adapt_state_dict_keys(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model.load_state_dict(new_state_dict)
    model.eval()

    # declare context and prediction windows:
    full_context_len = 64  # can be changed to any other number between config["min_seq_len"] and config["max_seq_len"]
    pred_len = 32  # can be changed to any other number between config["pred_len_min"] and config["pred_len"]
    periods1 = 123
    periods2 = 321
    periods = [periods1, periods2]

    eval_sine_cl_error(model, periods, full_context_len, pred_len, args.noise, device)


if __name__ == "__main__":
    main()
