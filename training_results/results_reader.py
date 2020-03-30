import os
from pathlib import Path

import pandas as pd
import numpy as np



def calculate_test_RMSE(y_true, y_pred):

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2))) 


def folders_in_directory(dir_path):
    dir_files = [os.path.join(dir_path,i) for i in os.listdir(dir_path)]
    folders = np.sort([i for i in dir_files if os.path.isdir(i) == True])
    return folders

def csv_in_directory(dir_path):
    csv_files = [os.path.join(dir_path,i) for i in os.listdir(dir_path) if '.csv' in i]
    return csv_files

csv_files = [i for i in os.listdir(os.getcwd()) if '.csv' in i]
for i in csv_files:
    os.remove(i)

for dataset_folder in folders_in_directory(os.getcwd()):
    dataset_name = os.path.basename(dataset_folder)
    
    for model_test_folder in folders_in_directory(dataset_folder):
        model_name = os.path.basename(model_test_folder)
        sample_name = dataset_name+'_'+model_name
        sample_dataframe = pd.DataFrame(columns=['best_rmse', 'rmse_mean', 'rmse_std', 'mean_time'])
        
        for sub_dataset_folder in folders_in_directory(model_test_folder):
            sub_dataset_name = os.path.basename(sub_dataset_folder)
            npz_files = np.sort(np.asarray([i for i in os.listdir(sub_dataset_folder) if '.npz' in i]))
            
            sub_dataset_res = []
            sub_dataset_train_time = []
            for npz_file in npz_files:
                npz = np.load(os.path.join(sub_dataset_folder, npz_file), allow_pickle=True)
                sub_dataset_train_time.append(npz['training_time'].astype(float))
                    
                y_true = npz['y_true']
                y_pred = npz['y_pred']
                
                if 'test_loss' not in npz.keys():
                    sub_dataset_res.append(calculate_test_RMSE(y_true, y_pred))
                else:
                    sub_dataset_res.append(npz['test_loss'])
                    
            best_rmse = '%.2f' % np.min(sub_dataset_res)                
            rmse_mean = '%.2f' % np.mean(sub_dataset_res)
            rmse_std = '%.4f' % np.std(sub_dataset_res)
            mean_time = '%2f' % np.mean(sub_dataset_train_time)

            data_dict = {'best_rmse':best_rmse, 'rmse_mean':rmse_mean, 'rmse_std':rmse_std, 'mean_time': mean_time}
            sample_dataframe = sample_dataframe.append(pd.DataFrame(data=data_dict, index=[sub_dataset_name]))
        sample_dataframe.to_csv(sample_name+'.csv', sep=';')