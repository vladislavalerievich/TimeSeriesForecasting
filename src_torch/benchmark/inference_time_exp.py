from chronos import ChronosPipeline
import torch
from gluonts.itertools import batcher
import tqdm
from typing import Iterable, Optional
import numpy as np
import argparse
import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from training.models import SSMModel, SSMModelMulti
from training.create_train_test_batch import GenerativeDataset, GenerativeDatasetMultiPoints
import yaml
from torch.utils.data import DataLoader
import time


#ssm config
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
model_name = "new_mi_4l_1024e_nores_30-512cl_m2_dconv_i5_lr1e-07_mp0.5_initlr1e-05_t300r_pm0.7_nPer_pl30_subday_subf"
model_string = f'../../models/{model_name}.pth'
sub_day = "subday" in model_name

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

def mambapfn_runner(dataset, cl, pred_len, device, batch_size=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSMModelMulti(scaler='min_max', sub_day=sub_day, **ssm_config).to(device)
    new_state_dict = adapt_state_dict_keys(torch.load(model_string, map_location=device)['model_state_dict'])
    model.load_state_dict(new_state_dict)
    model.eval()

    def scale_data(output, scaler):
        if scaler == 'custom_robust':
            output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
        elif scaler == 'min_max':
            output = (output['result'] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1)
        elif scaler == 'identity':
            output = output['result']
        return output

    def multipoint_predict(model, batch_size, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device):
        x = {}
        x['ts'] = batch_x_mark.to(device)
        x['history'] = batch_x.to(device)
        x['target_dates'] = batch_y_mark.to(device)
        x['task'] = torch.zeros(batch_size,pred_len).int().to(device)
        assert isinstance(model, SSMModelMulti), "Model must be an instance of SSMModelMulti"
        output = model(x, pred_len)
        output = scale_data(output, scaler)
        output = output.detach().cpu()
        return output

    inputs_tensor = dataset['history'][:, -cl:]
    inputs_tensor_ts = dataset['ts'][:, -cl:]
    inputs_tensor_task = dataset['task'][:, -cl:]
    inputs_tensor_target_dates = dataset['target_dates'][:, :pred_len]

    outputs = []
    start_time = time.time()
    with torch.no_grad():
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        for i, batch in enumerate(batcher(inputs_tensor, batch_size=batch_size)):
            batch_x = torch.stack(batch)
            batch_x_mark = inputs_tensor_ts[i*batch_x.shape[0]:(i+1)*batch_x.shape[0]]
            batch_y_mark = inputs_tensor_target_dates[i*batch_x.shape[0]:(i+1)*batch_x.shape[0]]
            
            outputs.append(multipoint_predict(model, batch_size, batch_x, batch_x_mark, batch_y_mark, pred_len, 'min_max', device))
        outputs = np.concatenate(outputs)
    end_time = time.time()
    print(f"Time taken for mambapfn with cl {cl} and pl {pred_len} with batch-size {batch_size} is {end_time-start_time} seconds")
    return end_time-start_time


def chronos_runner(dataset, cl, pred_len, model, device, batch_size=1):
    def generate_sample_forecasts(
        test_data_input: Iterable,
        pipeline: ChronosPipeline,
        prediction_length: int,
        batch_size: int,
        num_samples: int,
        **predict_kwargs,
    ):
        # Generate forecast samples
        forecast_samples = []
        # print(test_data_input.device)
        for batch in batcher(test_data_input, batch_size=batch_size): ## list of dictionaries of target(array) and start(datetime)
            context = torch.stack(batch)#.to(device)
            # print(context.shape)
            forecast_samples.append(
                pipeline.predict(
                    context, ## list of tensors of target values for each time series
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    **predict_kwargs,
                ).numpy()
            )
        forecast_samples = np.concatenate(forecast_samples)
        return forecast_samples

    if model == 'chronos-small':
        pipeline = ChronosPipeline.from_pretrained(
            'amazon/chronos-t5-small',
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    elif model == 'chronos-base':
        pipeline = ChronosPipeline.from_pretrained(
            'amazon/chronos-t5-base',
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    
    inputs_tensor = dataset['history'][:, -cl:]
    print(inputs_tensor.shape)

    start_time = time.time()
    with torch.no_grad():
        sample_forecasts = generate_sample_forecasts(
            inputs_tensor, 
            pipeline=pipeline,
            prediction_length=pred_len,
            batch_size=batch_size,
            num_samples=20,
            temperature=None,
            top_k=None, 
            top_p=None,
        )
        sample_forecasts = np.median(sample_forecasts,axis=1)
    end_time = time.time()
    print(f"Time taken for {model} with cl {cl} and pl {pred_len} with batch-size {batch_size} is {end_time-start_time} seconds")
    return end_time-start_time

def main_runner(models):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('../training/config.batch_ddp.yaml') as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    if not os.path.exists('../../data/inference_time_exp'):
        os.makedirs('../../data/inference_time_exp')
    if os.path.exists('../../data/inference_time_exp/inference_time_exp.yaml'):
        with open('../../data/inference_time_exp/inference_time_exp.yaml') as file:
            res_dict = yaml.load(file, yaml.loader.SafeLoader)
    else:
        res_dict = {}

    batch_size = 128
    cls = [512] #np.linspace(256, 2048, 8).astype(int)
    pls = [4] + np.linspace(8, 50, 10).astype(int).tolist()

    config["min_seq_len"] = max(cls)
    config["max_seq_len"] = max(cls)
    config["pred_len"] = max(pls)
    config["pred_len_sample"] = False
    config["batch_size"] = 2048
    config["sub_day"] = sub_day
    config["prior_mix_frac"] = 0.5
    val_dataset = GenerativeDatasetMultiPoints(config, cpus_available=1, device='cpu', mode='val', return_target_series=True)
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=val_dataset.worker_init_fn,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True,
    )
    prior_dataset = next(iter(val_data_loader))

    for model in models:
        if model not in res_dict.keys():
            res_dict[model] = {}
        for cl in cls:
            for pl in pls:

                if model == 'mambapfn':
                    time = mambapfn_runner(prior_dataset, cl, pl, device, batch_size)
                elif model.startswith('chronos'):
                    time = chronos_runner(prior_dataset, cl, int(pl), model, device, batch_size)

                key = f'{cl}_{pl}_{batch_size}'
                if key in res_dict[model].keys():
                    if isinstance(res_dict[model][key], float):
                        res_dict[model][key] = [res_dict[model][key]]
                else:
                    res_dict[model][key] = []
                res_dict[model][f'{cl}_{pl}_{batch_size}'].append(time)
        
                with open('../../data/inference_time_exp/inference_time_exp.yaml', 'w') as file:
                    yaml.dump(res_dict, file)
        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm")
    # args = parser.parse_args()
    models = ['chronos-base'] #, 'chronos-large', 'mambapfn']
    main_runner(models)