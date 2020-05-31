import os
import time
import numpy as np

from ANNModel import ANNModel
from CMAPSS_utils import CMAPSS_importer
from CMAPSS_utils import close_all
from utils.data_processing import MinMaxScaler


model_name = 'CNN'
dataset_name = 'CMAPSS'


results_dir =  os.path.join(os.getcwd(), 'training_results', dataset_name, model_name)
dataset_loader = CMAPSS_importer(dataset_name)
close_all()


from hyperopt import hp
search_space = {'cnn_layers': 4,
                'cnn_filters': 168, 
                'cnn_activation': 'relu',
                'cnn_kernel_height':  hp.quniform('cnn_kernel_height', 1,6,1),
                'cnn_kernel_width':  hp.quniform('cnn_kernel_width', 1,6,1),
                'cnn_padding': 'same',
              
                'hidden_layers': 2, 
                'layers_neurons': 200, 
                'layers_activation': 'relu',
                
                'dropout': 0.25,
                'LR': 0.001,
                'LR_patience': 5, 
                'ES_patience': 10}

# search_space = {'lstm_layers': hp.choice('lstm_layers', [1,2,3]),
#             'lstm_units': hp.quniform('lstm_units', 8, 256, 1), 
#             'lstm_activation': hp.choice('lstm_activation', ['relu']),
#             'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.75),
           
#             'hidden_layers': hp.choice('hidden_layers', [1,2,3]), 
#             'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1), 
#             'layers_activation': hp.choice('layers_activation', ['tanh', 'relu'])}


for sub_dataset in dataset_loader.subdatasets:
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.2)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)

    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  

    Model = ANNModel(x_train, y_train, x_valid, y_valid, x_test, y_test, model_type=model_name)     
    best, trials = Model.optimize(search_space, iters=250)
    print(best)