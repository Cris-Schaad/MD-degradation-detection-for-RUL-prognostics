import sys

sys.path.append('..')
from ANNModel import ANNModel
from FEMTO_utils import FEMTO_importer
from utils.data_processing import MinMaxScaler
from utils.training_functions import close_all


model_name = 'ConvLSTM'
dataset_loader = FEMTO_importer()
close_all()


data_names = dataset_loader.data_names
scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
from hyperopt import hp
search_space =  {'convlstm_layers': 1,
                'convlstm_filters': 16, 
                'convlstm_activation': 'relu',
                'convlstm_kernel_height': hp.quniform('convlstm_kernel_height', 1,6,1),
                'convlstm_kernel_width':  hp.quniform('convlstm_kernel_width', 1,6,1),
              
                'hidden_layers': 2, 
                'layers_neurons': 200, 
                'layers_activation': 'relu',
                
                'dropout': 0.25,
                'LR': 0.001,
                'LR_patience': 5, 
                'ES_patience': 10}


for sample, sample_name in enumerate([data_names[0]]):
    print('\n'+sample_name)
    x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.1)
    x_test, y_test = dataset_loader.get_test_sample(sample)
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  
            

    model = ANNModel(x_train, y_train, x_valid, y_valid, x_test=x_test, y_test=y_test, model_type = model_name)       
    best, trials = model.optimize(search_space, iters=100)
    print(best)