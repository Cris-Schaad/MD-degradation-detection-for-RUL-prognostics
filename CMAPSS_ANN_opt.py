import os
import time
import numpy as np


from models.OptimizableModel import OptimizableModel
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

# search_space = {'cnn_layers': hp.choice('cnn_layers', [1,2,3]),
#            'cnn_filters': hp.quniform('cnn_filters', 8, 256, 1), 
#            'cnn_activation': hp.choice('cnn_activation', ['tanh', 'relu']),
#            'cnn_kernel_height': hp.quniform('cnn_kernel_height', 1,5,1),
#            'cnn_kernel_width': hp.quniform('cnn_kernel_width', 1,5,1),
#            'cnn_padding': hp.choice('cnn_padding', ['valid', 'same']),
          
#            'hidden_layers': hp.choice('hidden_layers', [1,2,3]), 
#            'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1), 
#            'layers_activation': hp.choice('layers_activation', ['tanh', 'relu']),
#            'LRswitch_patience': hp.quniform('LRswitch_patience', 10, 50, 1), 
#            'EarlyS_patience': hp.quniform('EarlyS_patience', 10, 50, 1)}

search_space = {'lstm_layers': hp.choice('lstm_layers', [1,2,3]),
           'lstm_units': hp.quniform('lstm_units', 8, 256, 1), 
           'lstm_activation': hp.choice('lstm_activation', ['tanh', 'relu']),
           'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.75),
           
           'hidden_layers': hp.choice('hidden_layers', [1,2,3]), 
           'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1), 
           'layers_activation': hp.choice('layers_activation', ['tanh', 'relu']),
           'LRswitch_patience': hp.quniform('LRswitch_patience', 10, 50, 1), 
           'EarlyS_patience': hp.quniform('EarlyS_patience', 10, 50, 1)}


for sub_dataset in ['FD003']:
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.2)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  

    save_path = os.path.join(results_dir, sub_dataset)
    functions.save_folder(save_path)   
    
    model = OptimizableModel(x_train, y_train, x_valid, y_valid, x_test, y_test,
                             model_type = model_name)       
    best, trials = model.optimize(search_space, iters=100)
    print(best)