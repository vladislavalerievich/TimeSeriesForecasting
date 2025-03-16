"""
Module to train the model
"""
import sys
from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).parent

# Get the parent directory (project_root)
parent_dir = current_dir.parent

# Add the synthetic_generation directory to the sys.path
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "synthetic_generation"))

import os
import wandb
import torchmetrics
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import pprint
from models import SSMModel, SSMModelMulti, SSMModelNoPos
from create_train_test_batch import create_train_test_batch_dl
from real_data_val_pipeline import validate_on_real_dataset
from utils import SMAPEMetric, generate_model_save_name, avoid_constant_inputs
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


def train_model(config):
    print("config:")
    print(pprint.pformat(config))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'cuda device usage (before model load): {torch.cuda.memory_allocated() / 2**20}')

    # loading the datasets as dataloaders
    if config["debugging"]:
        available_cpus = 1
    else:
        available_cpus = os.cpu_count()

    base_model_configs = {
        "scaler": config['scaler'],
        "sin_pos_enc": config['sin_pos_enc'],
        "sin_pos_const": config['sin_pos_const'],
        "sub_day": config["sub_day"],
        "encoding_dropout": config["encoding_dropout"],
        "handle_constants_model": config["handle_constants_model"],
        }
    
    # Load the model
    if config['model_type'] == 'ssm':
        if config['no_pos_enc']:
            config['multipoint'] = True
            model = SSMModelNoPos(scaler=config['scaler'], **config['ssm_config']).to(device)
            print("Using SSMModelNoPos")
        else:
            if config['multipoint']:
                model = SSMModelMulti(**base_model_configs, **config['ssm_config']).to(device)
            else:
                model = SSMModel(**base_model_configs, **config['ssm_config']).to(device)
    else:
        raise ValueError('Model type not supported')

    # Assuming your train_loader and test_loader are already defined
    if config['lr_scheduler'] == "cosine":
        optimizer = optim.AdamW(model.parameters(), lr=config["initial_lr"])
        if config["t_max"] == -1:
            config["t_max"] = config['num_epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['learning_rate'])
    elif config['lr_scheduler'] == "cosine_warm_restarts":
        optimizer = optim.AdamW(model.parameters(), lr=config["initial_lr"])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['t_max'], eta_min=config['learning_rate'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    initial_epoch = 0
    # Load state dicts if we are resuming training
    config['model_save_name'] = generate_model_save_name(config)
    if config['continue_training'] and os.path.exists(f"{config['model_prefix']}/{config['model_save_name']}.pth"):
        print(f'loading previous training states from: {config["model_save_name"]}')
        ckpt = torch.load(f"{config['model_prefix']}/{config['model_save_name']}.pth", map_location=device)
        # load states
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if config['lr_scheduler'] == "cosine":
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        initial_epoch = ckpt['epoch']
    else:
        print('no previous training states found, starting fresh')
        model = model.to(device)
    
    
    train_dataloader, test_dataloader = create_train_test_batch_dl(config=config,
                                                                   initial_epoch=initial_epoch,
                                                                   cpus_available=available_cpus,
                                                                   device=device,
                                                                   multipoint=config['multipoint'])

    print(f'cuda device usage (after model load): {torch.cuda.memory_allocated() / 2**20}')
    config['model_param_size'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {config['model_param_size']}")
    #wandb hyperparam init
    if config["wandb"]:
        run = wandb.init(
            project="Time Series Forecasting",
            # Track hyperparameters and run metadata
            config=config,
            name=config['model_save_name']
        )

    
    criterion = nn.L1Loss().to(device) if (config["loss"] == "mae") else nn.MSELoss().to(device) # For Mean Squared Error Loss
    # Metric initialization
    train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_mse = torchmetrics.MeanSquaredError().to(device)
    train_smape = SMAPEMetric().to(device)
    
    val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_mse = torchmetrics.MeanSquaredError().to(device)
    val_smape = SMAPEMetric().to(device)
 
    print(f'cuda device usage (before training start): {torch.cuda.memory_allocated() / 2**20}') 

    # Training Loop 
    for epoch in range(initial_epoch, config['num_epochs']):
        print("============Training==================")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_epoch_loss = 0.0
        batch_idx = 0
        for batch_id, batch in enumerate(train_dataloader):
            data, target = {k: v.to(device) for k, v in batch.items() if k != 'target_values'}, batch['target_values'].to(device)           
            avoid_constant_inputs(data['history'], target)
            
            pred_len = target.size(1)
            optimizer.zero_grad()
            if isinstance(model, SSMModelMulti):
                drop_enc_allow = True
                if config["sample_multi_pred"] > np.random.rand():
                    drop_enc_allow = False
                    # randomly sample 2 numbers between 4 and pred_length (inclusive)
                    pred_limits = np.random.randint(4, pred_len+1, 2)
                    start_pred = min(pred_limits)
                    end_pred = max(pred_limits)

                    if end_pred == start_pred:
                        if start_pred == pred_len:
                            start_pred = start_pred - 1
                        else:
                            end_pred = end_pred + 1
                    pred_len = end_pred - start_pred
                    target = target[:, start_pred:end_pred].contiguous()
                    data['target_dates'] = data['target_dates'][:, start_pred:end_pred].contiguous()
                    data['complete_target'] = data['complete_target'][:, start_pred:end_pred].contiguous()
                    data['task'] = data['task'][:, start_pred:end_pred].contiguous()
                output = model(data, pred_len, training=True, drop_enc_allow=drop_enc_allow)
            elif isinstance(model, SSMModelNoPos):
                output = model(data, pred_len)
            else:
                output = model(data, training=True, drop_enc_allow=False)

            if config['scaler'] == 'min_max':
                max_scale = output['scale'][0].squeeze(-1)
                min_scale = output['scale'][1].squeeze(-1)
                scaled_target = (target - min_scale) / (max_scale - min_scale)
            else:                
                scaled_target = (target - output['scale'][0].squeeze(-1)) / output['scale'][1].squeeze(-1)
            
            loss = criterion(output['result'], scaled_target.float())
            loss.backward()
            
            # Log gradients
            if config["wandb"]:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()), "epoch": epoch})

            optimizer.step()

            torch.cuda.empty_cache()
            
            if config['scaler'] == 'min_max':
                inv_scaled_output = (output['result'] * (max_scale - min_scale)) + min_scale
            else:
                inv_scaled_output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)

            # Update metrics
            train_mape.update(inv_scaled_output, target)
            train_mse.update(inv_scaled_output, target)
            train_smape.update(inv_scaled_output, target)
            running_loss += loss.item()
            train_epoch_loss += loss.item()
            
            if batch_idx == config['training_rounds'] - 1:
                train_epoch_loss = running_loss / (batch_idx%10 + 1)

            if batch_idx % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Sc. Loss: {avg_loss} From torchmetric: {train_mse.compute()} From criterion: {loss.item()}')
                if config["wandb"]:    
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/mape": train_mape.compute(),
                        "train/mse": train_mse.compute(),
                        "train/smape": train_smape.compute(),
                        "step": epoch * config['training_rounds'] + batch_idx
                    })

                running_loss = 0.0
            
            batch_idx += 1
            #end of epoch at max training rounds
            if batch_idx == config['training_rounds']:
                break

        # Validation loop (after each epoch)
        print("============Validation==================")
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_batch_idx = 0
            for batch_id, batch in enumerate(test_dataloader):
                data, target = {k: v.to(device) for k, v in batch.items() if k != 'target_values'}, batch['target_values'].to(device)
                avoid_constant_inputs(data['history'], target)
                pred_len = target.size(1)
                if isinstance(model, SSMModelMulti) or isinstance(model, SSMModelNoPos):
                    output = model(data, pred_len)
                else:
                    output = model(data)
                
                if config['scaler'] == 'min_max':
                    max_scale = output['scale'][0].squeeze(-1)
                    min_scale = output['scale'][1].squeeze(-1)
                    scaled_target = (target - min_scale) / (max_scale - min_scale)
                else:                
                    scaled_target = (target - output['scale'][0].squeeze(-1)) / output['scale'][1].squeeze(-1)
                
                val_loss = criterion(output['result'], scaled_target.float()).item()
                total_val_loss += val_loss

                if batch_id % 10 == 9:
                    print(f'val loss for batch {batch_id}: {val_loss}')

                if config['scaler'] == 'min_max':
                    inv_scaled_output = (output['result'] * (max_scale - min_scale)) + min_scale
                else:
                    inv_scaled_output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
                # Update validation metrics
                val_mape.update(inv_scaled_output, target)
                val_mse.update(inv_scaled_output, target)
                val_smape.update(inv_scaled_output, target)

                if val_batch_idx == config['validation_rounds'] - 1:
                    break

        # Compute and log validation metrics
        avg_val_loss = total_val_loss / config['validation_rounds']
        print(f'Epoch: {epoch+1}, Sc. Validation Loss: {avg_val_loss} From torchmetric: {val_mse.compute()}')
        if config["wandb"]:   
            wandb.log({
                "val/loss": avg_val_loss,
                "val/mape": val_mape.compute(),
                "val/mse": val_mse.compute(),
                "val/smape": val_smape.compute(),
                "epoch": epoch
            }) 
     
        
        epoch_time = time.time() - epoch_start_time
        print(f'Time taken for epoch: {epoch_time/60} mins {epoch_time%60} secs.')

        # Reset metrics for the next epoch
        train_mape.reset()
        train_mse.reset()
        train_smape.reset()
        val_mape.reset()
        val_mse.reset()
        val_smape.reset()

        if epoch % config['real_test_interval'] == config['real_test_interval'] - 1:

                print("============Real Dataset Evaluation==================")

    # Initialize wandb Table
    table_data = []
    columns = ["Dataset", "MASE", "MAE", "RMSE", "SMAPE"]

    for real_dataset in config['real_test_datasets']:
        print(f'Evaluating on real dataset: {real_dataset}')
        
        # Run validation on the real dataset
        real_mase, real_mae, real_rmse, real_smape = validate_on_real_dataset(
            real_dataset, model, device, config['scaler'], subday=config["sub_day"]
        )
        
        print(f"MASE: {real_mase}, MAE: {real_mae}, RMSE: {real_rmse}, SMAPE: {real_smape}")

        # Append results to the table
        table_data.append([real_dataset, real_mase, real_mae, real_rmse, real_smape])

    if config["wandb"]:
        wandb.log({"Real Dataset Metrics": wandb.Table(data=table_data, columns=columns), "epoch": epoch})
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch})
        
        if config['lr_scheduler'].startswith("cosine"):
            if (scheduler.get_last_lr()[0] == config['learning_rate']) & (config['lr_scheduler'] == "cosine"):
                print("Learning rate has reached the minimum value. No more steps.")
            else:
                scheduler.step()
            
        if epoch % 5 == 4:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if config['lr_scheduler'] == "cosine" else None,
                'epoch': epoch,
            }
            torch.save(ckpt, f"{config['model_prefix']}/{config['model_save_name']}.pth")

    if config["wandb"]:
        wandb.finish()

    # Save the final model
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if config['lr_scheduler'] == "cosine" else None,
        'epoch': epoch,
    }
    torch.save(ckpt, f"{config['model_prefix']}/{config['model_save_name']}_Final.pth")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.batch_ddp.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)
    
    # for wandb offline mode (can comment if needed):
    os.environ['WANDB_MODE'] = "online" if config['wandb'] else 'offline'
    train_model(config)
