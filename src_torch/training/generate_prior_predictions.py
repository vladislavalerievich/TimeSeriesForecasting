import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import yaml
import tqdm
import os
import sys
import random

sys.path.append('../')
from training.models import SSMModel, SSMModelMulti
from training.create_train_test_batch import GenerativeDataset
from torch.utils.data import DataLoader

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    assert isinstance(model, SSMModelMulti), "Model must be an instance of SSMModelMulti"
    output = model(x, pred_len)
    output = scale_data(output, scaler)
    output = output.detach().cpu().squeeze()
    return output

with open('config.batch_ddp.yaml') as config_file:
    config = yaml.load(config_file, yaml.loader.SafeLoader)

val_dataset = GenerativeDataset(config, cpus_available=8, device='cpu', mode='val', return_target_series=True)
val_data_loader = DataLoader(
    dataset=val_dataset,
    batch_size=None,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
    worker_init_fn=val_dataset.worker_init_fn,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True,
)

ssm_config = {
"bidirectional":False,
"conv_d":4,
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

model_path = '../../models/ssm_mi_anneal_min_max_2l_1024e_nores_512cl_fullcl_lr1e-08_conv30_initlr1e-05_t600.pth'
model_string = model_path.split("/")[-1].replace(".pth","")

x = next(iter(val_data_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SSMModel(scaler='min_max', **ssm_config).to(device)

print(f"Loading model {model}")

# print(f"loaded model is {torch.load(model_path, map_location=device)['model_state_dict']}")
model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
model.eval()

x['target_values'] = x['target_series'][:,:,5].squeeze()
x['target_dates'] = x['target_series'][:,:,:5]

pred_dfs = []

for i in range(x['history'].shape[0]):
    id = [str(i)]
    y = multipoint_predict(model, x['history'][i].unsqueeze(0), x['ts'][i].unsqueeze(0), x['target_dates'][i].unsqueeze(0), 24, 'min_max', device)
    hist_df = pd.DataFrame({'id': id*x['history'].shape[1], 'target':x['history'][i], 'pred':x['history'][i]})
    pred_df = pd.DataFrame({'id': id*24, 'pred':y, 'target':x['target_values'][i]})
    pred_dfs.append(pd.concat([hist_df,pred_df], axis=0))

pred_df = pd.concat(pred_dfs)

if not os.path.exists(f'../../data/prior_preds'):
    os.makedirs(f'../../data/prior_preds')
if not os.path.exists(f'../../data/prior_preds/{model_string}'):
    os.makedirs(f'../../data/prior_preds/{model_string}')

with open(f'../../data/prior_preds/{model_string}/prior_predictions.pkl', 'wb') as f:
    pickle.dump(pred_df, f)