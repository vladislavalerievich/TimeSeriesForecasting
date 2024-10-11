from hmac import new
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import yaml
import argparse
import sys
from datetime import date
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta
import os

sys.path.append('../')
from training.models import SSMModel, SSMModelMulti
from training.create_train_test_batch import GenerativeDataset
from synthetic_generation.constants import *
from training.utils import save_figure_for_latex
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    y = np.stack([y_sin1, y_sin1+y_lin, y_sin1 * y_sin2, y_sin1 * y_lin], axis=0)

    if noise:
        y += np.random.normal(0, 0.05, N)
    
    return y


def scale_data(output, scaler):
    if scaler == 'custom_robust':
        output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
    elif scaler == 'min_max':
        output = (output['result'] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1)
    elif scaler == 'identity':
        output = output['result']
    return output

def batch_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.repeat(eval_pred_len, 1, 1).to(device)
    x['history'] = batch_x.repeat(eval_pred_len, 1).to(device)

    x['target_dates'] = batch_y_mark.transpose(0, 1).to(device)
    x['task'] = torch.zeros(eval_pred_len,1).int().to(device)

    output = model(x)
    output = scale_data(output, scaler)
    
    output = output.detach().cpu().squeeze()
    
    return output

def multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(4,pred_len).int().to(device)
    assert isinstance(model, SSMModelMulti), "Model must be an instance of SSMModelMulti"
    output = model(x, pred_len)
    output = scale_data(output, scaler)
    output = output.detach().cpu()
    return output

ssm_config = {
    "bidirectional":False,
    "enc_conv" : True,
    "init_dil_conv" : True,
    "enc_conv_kernel" : 5,
    "init_conv_kernel" : 5,
    "init_conv_max_dilation" : 3,
    "global_residual":False,
    "in_proj_norm":False,
    "initial_gelu_flag":True,
    "linear_seq":15,
    "mamba2":True,
    "norm":True,
    "norm_type":"layernorm",
    "num_encoder_layers":2,
    "residual":False,
    "token_embed_len":1024,
}

model_path = '../../models/latest_mi_2l_1024e_nores_30-512cl_m2_norm_dconv_i5e5_lr1e-07_mp0.5_initlr1e-05_t300r_pm0.7_nPer_pl10-60_subday_subfreq0.2_perfreq0.5_d0.05_s0.05.pth'
model_string = model_path.split("/")[-1].replace(".pth","")
subday = 'subday' in model_string

def eval_sine_cl_error(model, periods, full_context_len, pred_len, noise, device):
    # Generate a sine wave
    periods = periods
    seq_len = full_context_len + pred_len
    x = generate_sine_waves(seq_len, periods[0], periods[1], noise=noise)

    # generate time stamps
    freq = freq_dict[np.random.choice(['daily', 'weekly', 'monthly'])]['freq']
    start = pd.Timestamp(date.fromordinal(int((BASE_START - BASE_END)*beta.rvs(5,1)+BASE_START)))
    ts =  pd.date_range(start=start, periods=seq_len, freq=to_offset(freq))

    ts = np.stack([ts.year.values, 
            ts.month.values, 
            ts.day.values, 
            ts.day_of_week.values + 1, 
            ts.day_of_year.values], axis=-1)

    mses = {}

    for context_len in range(40+pred_len,full_context_len):

        # Create a batch
        batch = {}
        batch['ts'] = torch.tensor(ts[-context_len:-pred_len]).unsqueeze(0)
        batch['history'] = torch.tensor(x[-context_len:-pred_len]).unsqueeze(0)

        batch['target_dates'] = torch.tensor(ts[full_context_len:]).unsqueeze(0)

        output = multipoint_predict(model, batch['history'], batch['ts'], batch['target_dates'], pred_len, 'min_max', device)

        mse = np.mean((output - x[-pred_len:])**2)
        mae = np.abs(output - x[-pred_len:]).mean()
        mses[context_len] = {'mse': mse, 'mae': mae}

    mse_df = pd.DataFrame.from_dict(mses, orient='index', columns=['mse', 'mae'])

    with open(f'../../data/sine_error_{pred_len}_{full_context_len}_{periods}pi.pkl', 'wb') as f:
        pickle.dump(mse_df, f)


def eval_sine_preds(model, periods, full_context_len, pred_len, noise, device):

    # Generate a sine wave
    periods = periods
    seq_len = full_context_len + pred_len
    x = generate_sine_waves(seq_len, periods[0], periods[1], noise=noise)

    # generate time stamps
    freq = freq_dict['weekly']['freq']
    start = pd.Timestamp(date.fromordinal(int((BASE_START - BASE_END)*beta.rvs(5,1)+BASE_START)))
    ts_pd =  pd.date_range(start=start, periods=seq_len, freq=to_offset(freq))

    if not subday:
        ts = np.stack([ts_pd.year.values, 
                ts_pd.month.values, 
                ts_pd.day.values, 
                ts_pd.day_of_week.values + 1, 
                ts_pd.day_of_year.values], axis=-1)
    else:
        ts = np.stack([
                ts_pd.year.values, 
                ts_pd.month.values, 
                ts_pd.day.values, 
                ts_pd.day_of_week.values + 1, 
                ts_pd.day_of_year.values,
                ts_pd.hour.values,
                ts_pd.minute.values], axis=-1)

    pred_dfs = []

    # contexts_to_eval = np.linspace(0,full_context_len-1,9).astype(int)[1:]
    contexts_to_eval = [16,32,128,256,512]

    for context_len in contexts_to_eval:
        id = [str(context_len)]
        # Create a batch
        batch = {}
        batch['ts'] = torch.tensor(ts[-(context_len+pred_len):-pred_len]).unsqueeze(0).repeat(4,1,1)
        batch['history'] = torch.tensor(x[:, -(context_len+pred_len):-pred_len])

        batch['target_dates'] = torch.tensor(ts[full_context_len:]).unsqueeze(0).repeat(4,1,1)

        output = multipoint_predict(model, batch['history'], batch['ts'], batch['target_dates'], pred_len, 'min_max', device)
        

        hist_df = pd.DataFrame({'id': id*batch['history'].shape[1],
                                'ts':ts_pd[-(context_len+pred_len):-pred_len],
                                'target_sin':x[0][-(context_len+pred_len):-pred_len],
                                'target_sinal':x[1][-(context_len+pred_len):-pred_len],
                                'target_sinms':x[2][-(context_len+pred_len):-pred_len],
                                'target_sinml':x[3][-(context_len+pred_len):-pred_len],
                                'pred_sin':x[0][-(context_len+pred_len):-pred_len],
                                'pred_sinal':x[1][-(context_len+pred_len):-pred_len],
                                'pred_sinms':x[2][-(context_len+pred_len):-pred_len],
                                'pred_sinml':x[3][-(context_len+pred_len):-pred_len]})
        pred_df = pd.DataFrame({'id': id*pred_len,
                                'ts':ts_pd[-pred_len:],
                                'target_sin':x[0][-pred_len:],
                                'target_sinal':x[1][-pred_len:],
                                'target_sinms':x[2][-pred_len:],
                                'target_sinml':x[3][-pred_len:],
                                'pred_sin':output[0],
                                'pred_sinal':output[1],
                                'pred_sinms':output[2],
                                'pred_sinml':output[3]})
        pred_dfs.append(pd.concat([hist_df,pred_df], axis=0))

    pred_df = pd.concat(pred_dfs)

    # create a folder with the name 'sine_exp' inside the data folder and then a subfolder with the model string (if they dont already exist)
    if not os.path.exists(f'../../data/sine_exp'):
        os.makedirs(f'../../data/sine_exp')
    if not os.path.exists(f'../../data/sine_exp/{model_string}'):
        os.makedirs(f'../../data/sine_exp/{model_string}')

    # saving the image of some predictions
    cls = contexts_to_eval # can edit this to change the context lengths to plot
    # reverse the order of the context lengths to plot
    cls = cls[::-1]
    plots = ['sin', 'sinal', 'sinms', 'sinml']
    fig, axes = plt.subplots(2, 2, figsize=(24, 8))
    plt.rcParams["figure.autolayout"] = True
    colors = ['maroon', 'magenta', 'green', 'orange', 'grey']
        
    for i, plot in enumerate(plots):
        ax = axes.flatten()[i]
        sns.lineplot(data=pred_df[pred_df['id'] == str(cls[0])], x='ts', y=f'target_{plot}', ax=ax, c='blue')
        ax.axvline(x = pred_df[pred_df['id'] == str(cls[0])]['ts'].values[cls[0]], color='black', linestyle='solid', linewidth=1.5)
        for j, cl in enumerate(cls):
            data = pred_df[pred_df['id'] == str(cl)].reset_index(drop=True)
            min_ts = data['ts'].min()
            sns.lineplot(data=data, x='ts', y=f'pred_{plot}', ax=ax, c=colors[j])
            ax.axvline(x=min_ts, color=colors[j], linestyle='dashed')
            ax.grid(True)
            plt.tight_layout()
        if i == 0:
            ax.set_ylabel('Function Value', fontsize=20)
            ax.set_xlabel(None)
        elif i == 1:
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        elif i == 2:
            ax.set_ylabel('Function Value', fontsize=20)
            ax.set_xlabel('Time Step', fontsize=20)
        elif i == 3:
            ax.set_xlabel('Time Step', fontsize=20)
            ax.set_ylabel(None)
        ax.legend()
    
    fig_savename = f'../../data/sine_exp/{model_string}/pred_img{pred_len}_{full_context_len}_{periods[0]}pi'
    save_figure_for_latex(fig=fig, filename=fig_savename)

    with open(f'../../data/sine_exp/{model_string}/{pred_len}_{full_context_len}_{periods[0]}pi_am.pkl', 'wb') as f:
        pickle.dump(pred_df, f)


def adapt_state_dict_keys(old_state_dict):
    new_state_dict = {}

    for key in old_state_dict.keys():
        if "linear_layer" in key:
            # Extract the layer index
            layer_idx = key.split('.')[1]
            
            # Replace "linear_layer" with "stage_2_layer.0"
            new_key = key.replace(f"linear_layer", f"stage_2_layer.0")
            
            # Add the updated key to the new state dict
            new_state_dict[new_key] = old_state_dict[key]
        else:
            # Keep other keys unchanged
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSMModelMulti(scaler='min_max', sub_day=subday, **ssm_config).to(device)
    new_state_dict = adapt_state_dict_keys(torch.load(model_path, map_location=device)['model_state_dict'])
    model.load_state_dict(new_state_dict)
    model.eval()

    print(args)
    # declare context and prediction windows:
    full_context_len = args.cl
    pred_len = args.pl
    periods1 = args.periods1
    periods2 = args.periods2
    periods = [periods1, periods2]
    exp = args.experiment

    if exp == 'error':
        eval_sine_cl_error(model, periods, full_context_len, pred_len, args.noise, device)
    elif exp == 'preds':
        eval_sine_preds(model, periods, full_context_len, pred_len, args.noise, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cl", type=int, default=512, help="the full context length")
    parser.add_argument("-p", "--pl", type=int, default=36, help="the prediction length")
    parser.add_argument("-pi1", "--periods1", type=int, default=16, help="the number of periods in 1st sine wave")
    parser.add_argument("-pi2", "--periods2", type=int, default=24, help="the number of periods in 2nd sine wave")
    parser.add_argument("-e", "--experiment", type=str, default='preds', help="the experiment to run")
    parser.add_argument("-n", "--noise", type=str, default=True, help="the experiment to run")
    args = parser.parse_args()

    main(args)