import os
import time
import numpy as np


from models.LSTM import CMAPSSBayesianOptimizer
from utils import dataset_importer
import utils.training_functions as functions


model_name = 'LSTM'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'C-MAPSS', model_name)
dataset_dir = os.path.join(os.getcwd(), 'data', 'C-MAPSS', 'processed_data', 'CMAPSS_dataset.npz')
dataset_npz = dict(np.load(dataset_dir, allow_pickle=True))['dataset'][()]
dataset_loader = dataset_importer.CMAPSS_importer(dataset_dir)
functions.plt_close()

scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=2)
scaler_y = functions.MinMaxScaler(feature_range=(0,1))

for sub_dataset in ['FD003']:
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.2)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  

    save_path = os.path.join(results_dir, sub_dataset)
    functions.save_folder(save_path)   
    
    model = CMAPSSBayesianOptimizer(x_train, y_train, x_valid, y_valid, x_test, y_test)       
    best, trials = model.optimize(iters=100)
    
    print(best)