import torch
import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import yaml
import argparse
import submitit
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from benchmark.data_provider.data_factory import data_provider
from utilsforecast.losses import mase, mae, smape, rmse
import csv
from training.models import SSMModel, SSMModelMulti
from training.utils import save_figure_for_latex
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

REAL_DATASETS = {
    "nn5_daily_without_missing": 56,
    "nn5_weekly": 8,
    "covid_deaths": 30,
    "weather": 30,
    "hospital": 12, 
    "fred_md": 12,
    "car_parts_without_missing": 12,
    "traffic": 24,
    "m3_monthly": 18,
    "ercot": 24,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    "cif_2016": 12,
    "exchange_rate": 30,
    "m3_quarterly": 8,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
}

MAX_LENGTH = 512

ssm_config = {
    "bidirectional":False,
    "enc_conv" : False,
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
    "num_encoder_layers":4,
    "residual":False,
    "token_embed_len":1024,
}

model_path = '../../models/new_mi_4l_1024e_nores_30-512cl_m2_dconv_i5_lr1e-07_mp0.5_initlr1e-05_t300r_pm0.4_nPer_pl10-60_subday.pth'
model_string = model_path.split("/")[-1].replace(".pth","")
subday = 'subday' in model_string

def set_queue(q_, log_folder, maximum_runtime=None):
    global ex
    global q
    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if q_ == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if maximum_runtime is None:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*4-1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        tasks_per_node=1,
                        nodes=1,
                        cpus_per_task=30, #24
                        mem_per_cpu=4096,
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime

def scale_data(output, scaler):
    if scaler == 'custom_robust':
        output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
    elif scaler == 'min_max':
        output = (output['result'] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1)
    elif scaler == 'identity':
        output = output['result']
    return output

def auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, eval_pred_len, real_data_args, scaler, device):
        # decoder input
        dec_inp = torch.zeros_like(
            batch_y[-eval_pred_len:]).unsqueeze(0).int().to(device)

        x = {}
        x['ts'] = batch_x_mark.to(device)
        x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
        outputs = []
        for pred_ind in range(0, eval_pred_len):
            x['target_dates'] = batch_y_mark[:, pred_ind].unsqueeze(1).to(device)
            x['task'] = dec_inp[:, pred_ind].unsqueeze(1)
            multipoint = isinstance(model, SSMModelMulti)
            if multipoint:
                output = model(x, 1)
            else:
                output = model(x)
            output = scale_data(output, scaler)
            outputs.append(output)
            
            x['history'] = torch.cat([x['history'], output.unsqueeze(0)], dim=1)
            x['ts'] = torch.cat([x['ts'], x['target_dates']], dim=1)
    
        outputs = torch.stack(outputs, dim=1).detach().cpu().squeeze()
            
        return outputs
    

def batch_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.repeat(eval_pred_len, 1, 1).to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).repeat(eval_pred_len, 1).to(device)

    x['target_dates'] = batch_y_mark.transpose(0, 1).to(device)
    x['task'] = torch.zeros(eval_pred_len,1).int().to(device)

    output = model(x)
    output = scale_data(output, scaler)
    
    output = output.detach().cpu().squeeze()
    
    return output

def multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    assert isinstance(model, SSMModelMulti), "Model must be an instance of SSMModelMulti"
    output = model(x, pred_len)
    output = scale_data(output, scaler)
    output = output.detach().cpu().squeeze()
    return output


def ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device, ensemble_config):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    
    cl = x["history"].size(1)
    
    drops = np.linspace(0,cl*ensemble_config["max_fraction"], ensemble_config["num_ensembles"]).astype(int)
    assert isinstance(model, SSMModelMulti), "Model must be an instance of SSMModelMulti"
  
    outputs = []
    z = {}
    for drop in drops:
        drop_indices = np.random.choice(cl, size=drop, replace=False)  # Randomly choose 'drop' indices to remove
        keep_indices = np.setdiff1d(np.arange(cl), drop_indices)
        z['ts'] = x['ts'][:, keep_indices, :]
        z['history'] = x['history'][:, keep_indices]
        z['target_dates'] = x['target_dates']
        z['task'] = x['task']
        output = model(x, pred_len)
        output = scale_data(output, scaler)
        output = output.detach().cpu().squeeze()
        outputs.append(output)
    outputs = torch.stack(outputs, dim=1)
    if ensemble_config["method"] == "median":
        output = outputs.median(dim=1)
    else:
        output = outputs.mean(dim=1)
    output = output.detach().cpu().squeeze()
    return output


def evaluate_real_dataset(dataset: str, model, scaler, context_len, eval_pred_len, device, pred_style=None):

    with open('./real_data_args.yaml') as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if pred_style is None:
        pred_style = real_data_args['pred_style']
        
    if real_data_args['pad']:
        real_data_args['data_path'] = dataset + f'_pad_{MAX_LENGTH}.pkl'
    else:
        real_data_args['data_path'] = dataset + f'_nopad_{MAX_LENGTH}.pkl'

    pred_len = REAL_DATASETS[dataset]
    real_data_args['data'] = dataset
    real_data_args['pred_len'] = pred_len
    test_dataset, test_dataloader = data_provider(real_data_args, real_data_args['flag'], subday=subday)

    gts_dataset = get_dataset(real_data_args['data'], regenerate=False)
    seasonality = get_seasonality(gts_dataset.metadata.freq)
    print(seasonality)

    batch_train_dfs = []
    batch_pred_dfs = []
    model.eval()
    j = 0
    print(f'pred_style: {pred_style}')
    with torch.no_grad():
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        for i, batch in tqdm(enumerate(test_dataloader)):
            cl = min(context_len, batch['x'].shape[1])
            if cl < context_len:
                continue
            ids = batch["id"]
            batch_x = batch["x"][:, -cl:, :].float()
            batch_y = batch["y"][:, :eval_pred_len, :].squeeze().float()

            batch_x_mark = batch["ts_x"][:, -cl:, :].float()
            batch_y_mark = batch["ts_y"][:, :eval_pred_len, :].float()
            
            if pred_style == 'multipoint':
                output = multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)
            elif pred_style == 'auto_regressive':
                output = auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, eval_pred_len, real_data_args, scaler, device)
            elif pred_style == 'ensemble':
                output = ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device, real_data_args['ensemble_config'])
            else:
                output = batch_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)
            
            # create dfs used to calculate the MASE such that the batch['x'] and pred are squeezed into 1 column and the ids are repeated 
            # for batch['x'].shape[1] times and pred_len times respectively
            batch_train_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(batch_x.size(1)).numpy(),
                'target': batch_x.flatten().numpy()
            }))
            
            batch_pred_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(eval_pred_len).numpy(),
                'pred': output.flatten().numpy(),
                'target': batch_y.flatten().numpy()
            }))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    mase_loss = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
    mae_loss = mae(pred_df, ['pred'], 'id', 'target')
    rmse_loss = rmse(pred_df, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df, ['pred'], 'id', 'target')

    out_dict = {'mase': mase_loss['pred'].mean(), 'mae': mae_loss['pred'].mean(), 'rmse': rmse_loss['pred'].mean(), 'smape': smape_loss['pred'].mean()}

    return out_dict, train_df, pred_df


def csv_file_writer(res_dict, filename):
    all_fieldnames = set()
    for entry in res_dict.values():
        all_fieldnames.update(entry.keys())
    # Convert set to list and sort or maintain any specific order if required
    fieldnames = list(all_fieldnames)
    fieldnames.insert(0, 'ID')
    # Specify the file name

    # Open the file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a DictWriter object with the extended fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the data
        for key, value in res_dict.items():
            # Insert the outer key into the row dictionary
            row = {'ID': key}
            row.update(value)
            writer.writerow(row)


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


def plot_mase_over_cl(filename, dataset_name, model_string, dataset_pl, pred_style):
    
    with open(filename, 'r') as file:
        data = yaml.load(file, yaml.Loader)

    data = pd.DataFrame(data).transpose().reset_index()
    data['cl'] = data['index'].apply(lambda x: int(x.split('_')[0]))
    data['pl'] = data['index'].apply(lambda x: int(x.split('_')[1]))
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

    # First plot: MASE over different Context Lengths for Prediction Length 24
    #plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.lineplot(data[data['pl'] == dataset_pl].set_index('cl')['mase'], ax=axes[0])
    axes[0].set_xlabel('Context Length', fontsize=18)
    axes[0].set_xlim(data['cl'].min(), data['cl'].max())
    axes[0].set_ylabel('MASE', fontsize=18)
    axes[0].set_title('(a)', fontsize=18)
    axes[0].legend()
    axes[0].grid(True)

    # Second plot: MASE over different prediction Lengths
    #plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    print(data['cl'].unique())
    sns.lineplot(data=data[data['cl'] == data['cl'].max()], x='pl', y='mase', ax=axes[1])
    axes[1].set_xlabel('Prediction Length', fontsize=18)
    axes[1].set_ylabel('MASE', fontsize=18)
    axes[1].set_xlim(data['pl'].min(), data['pl'].max())
    axes[1].set_title('(b)', fontsize=18)
    axes[1].legend()
    axes[1].grid(True)

    # Save the figure as an image file (e.g., PDF format)
    fig_savename = f'../../data/real_data_evals/{model_string}/{pred_style}/{dataset_name}_mase_cl_pl'
    save_figure_for_latex(fig, fig_savename)

def main_evaluator(pred_style=None):
    
    with open('./real_data_args.yaml') as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if pred_style is None:
        pred_style = real_data_args['pred_style']

    if not os.path.exists(f'../../data/real_data_evals/{model_string}/{pred_style}'):
        os.mkdir(f'../../data/real_data_evals/{model_string}/{pred_style}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSMModelMulti(scaler='min_max', sub_day=subday, **ssm_config).to(device)
    new_state_dict = adapt_state_dict_keys(torch.load(model_path, map_location=device)['model_state_dict'])
    model.load_state_dict(new_state_dict)
    
    context_lens = np.linspace(4, MAX_LENGTH, 50).astype(int)
    print(context_lens)

    for dataset in [i for i in os.listdir('../../data/real_val_datasets') if i.endswith(f'ercot_nopad_{MAX_LENGTH}.pkl')]:
        dataset_name = ('_').join(dataset.split('_')[:-2])
        print(dataset)
        res_dict = {}
        print(f'pred_style: {pred_style}')
        if not os.path.exists(f'../../data/real_data_evals/{model_string}/{pred_style}/{dataset_name}_all_cl_pl.yml'):
            pred_lens = range(1, REAL_DATASETS[dataset_name]+1)
            for cl in context_lens:
                for pl in pred_lens:
                    print(f'evaluating {dataset_name} for context length:{cl} and prediction length:{pl}')
                    out_dict, train_df, pred_df = evaluate_real_dataset(dataset_name, model, 'min_max', cl, pl, device, pred_style=pred_style)
                    res_dict[f'{cl}_{pl}'] = out_dict

            with open(f'../../data/real_data_evals/{model_string}/{pred_style}/{dataset_name}_all_cl_pl.yml', 'w') as outfile:
                yaml.dump(res_dict, outfile, default_flow_style=True)
            
            filename = f'../../data/real_data_evals/{model_string}/{pred_style}/{dataset_name}_all_cl_pl.csv'
            csv_file_writer(res_dict, filename)
                
        plot_mase_over_cl(f'../../data/real_data_evals/{model_string}/{pred_style}/{dataset_name}_all_cl_pl.yml',
                          dataset_name, model_string, REAL_DATASETS[dataset_name], pred_style)
            
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm")
    args = parser.parse_args()

    directory = f'../../data/real_data_evals/{model_string}'
    if not os.path.exists(directory):
        os.mkdir(directory)
    print(args.slurm)
    if args.slurm:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = '../logs/'
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main_evaluator)

        print(job)
    else:
        print("Running on local machine")
        for pred_style in ['multipoint']:#, 'ensemble']:
            main_evaluator(pred_style)


    