# Install AutoGluon if not already installed (uncomment the line below)
# !pip install autogluon
import sys
import os
sys.path.append('..') 
from tkinter.tix import MAX
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from data_provider.data_factory import data_provider
import yaml
from gluonts.dataset.repository.datasets import get_dataset
import pickle
import argparse
import submitit

MAX_LENGTH = 2048

REAL_DATASETS = {
    "nn5_daily_without_missing": 56,
    "nn5_weekly": 8,
    "covid_deaths": 30,
    "weather": 30,
    "hospital": 12, 
    "fred_md": 12,
    "car_parts_without_missing": 12,
    # "traffic": 24, ## autoarima gets stuck
    "m3_monthly": 18,
    "ercot": 24,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    # "cif_2016": 12,
    "electricity_hourly": 24,
    "m4_daily": 14,
    "exchange_rate": 30,
    "m3_quarterly": 8,
    "m3_yearly": 6,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
}

def gluonts_to_dataframe(gluonts_dataset):
    data_list = []
    for entry in gluonts_dataset:
        target = entry['target']
        start_date = entry['start']
        time_index = pd.date_range(start=start_date.to_timestamp(), periods=len(target), freq=start_date.freq)
        
        item_df = pd.DataFrame({
            'item_id': entry['item_id'],
            'timestamp': time_index,
            'target': target
        })
        data_list.append(item_df)
    
    return pd.concat(data_list).reset_index(drop=True)

def main():
    if os.path.exists('../../data/autogluonts_evaluation_results/autogluonts_metrics.pkl'):
        with open('../../data/autogluonts_evaluation_results/autogluonts_metrics.pkl', 'rb') as f:
            model_results = pickle.load(f)
        print(f'loaded model results: \n{model_results}')
    else:
        model_results = {}

    for model in ["Chronos_small_mod", "Chronos_base_mod"]: #'AutoARIMA', 'DeepAR'
        print(f"Running AutoGluon with {model} model...")
        dataset_metrics = {}
        for dataset_name, prediction_length in REAL_DATASETS.items():
            print(f"Running AutoGluon on {dataset_name} dataset...")
            # Step 1: Load the GluonTS dataset
            dataset = get_dataset(dataset_name)

            # Access the train and test datasets from GluonTS
            gluonts_train_data = dataset.train
            gluonts_test_data = dataset.test

            # Convert train and test datasets to pandas DataFrame
            train_df = gluonts_to_dataframe(gluonts_train_data)
            test_df = gluonts_to_dataframe(gluonts_test_data)

                    # Step 3: Convert DataFrames to AutoGluon TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_df, 
                id_column='item_id', 
                timestamp_column='timestamp'
            ).slice_by_timestep(-(MAX_LENGTH),None)

            test_data = TimeSeriesDataFrame.from_data_frame(
                test_df, 
                id_column='item_id', 
                timestamp_column='timestamp'
            ).slice_by_timestep(-(MAX_LENGTH + prediction_length),None)

            # Step 4: Train AutoGluon Model using the GluonTS `train` data
            predictor = TimeSeriesPredictor(
                prediction_length=int(prediction_length), 
                target='target',
                eval_metric='MASE'  # You can choose other metrics like 'MAPE', 'sMAPE', etc.
            )

            if model.startswith('Chronos'):
                model_hps = {model.split('_')[0]: {'model_path': model.split('_')[1], 'batch_size': 16, 'device': 'cuda:0'}}
            else:
                model_hps = {model: {}}  # Example models: AutoARIMA and DeepAR
            
            predictor.fit(
                train_data=train_data, 
                enable_ensemble=False,
                hyperparameters=model_hps,  # Example models: AutoARIMA and DeepAR
                #time_limit=600,
            )

            # Step 5: Make predictions using the trained model on the `train_data`
            predictions = predictor.predict(test_data)

            # Step 6: Evaluate the model using the actual `test_data`
            evaluation = predictor.evaluate(test_data)

            # Step 7: Output the evaluation results
            print("Evaluation Metrics:", evaluation)
            
            with open(f'../../data/autogluonts_evaluation_results/{model}_{dataset_name}_predictions.pkl', 'wb') as f:
                pickle.dump(predictions, f)
            
            dataset_metrics[dataset_name] = evaluation["MASE"]
        
        model_results[model] = dataset_metrics
        
    with open('../../data/autogluonts_evaluation_results/autogluonts_metrics.pkl', 'wb') as f:
        pickle.dump(model_results, f)

    pd.DataFrame(model_results).to_csv('../../data/autogluonts_evaluation_results/autogluonts_metrics.csv')    


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
                        cpus_per_task=4, #24
                        mem_per_cpu=4096,
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm") # if not provided you can just run it from terminal (for debugging)
    args = parser.parse_args()

    if args.slurm == True:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = '../logs/'
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main)

        print(job)
    else:
        print("Running on local machine")
        print(args.slurm)
        main()
