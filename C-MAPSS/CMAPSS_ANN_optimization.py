import os
import sys

sys.path.append('..')
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



for sub_dataset in ['FD001']:
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.2)

    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)

    Model = ANNModel(x_train, y_train, x_valid, y_valid, model_type=model_name)     
    best, trials = Model.optimize(search_space, iters=100)
    print(best)