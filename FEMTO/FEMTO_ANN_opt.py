import os

from models.OptimizableModel import OptimizableModel
from utils import dataset_importer
import utils.training_functions as functions


model_name = 'ConvLSTM'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'FEMTO', model_name)
dataset_dir = os.path.join(os.getcwd(), 'data', 'FEMTO', 'processed_data', 'FEMTO_dataset.npz')
dataset_loader = dataset_importer.FEMTO_importer(dataset_dir)
functions.plt_close()

data_names = dataset_loader.data_names
scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=2)
scaler_y = functions.MinMaxScaler(feature_range=(0,1))


from hyperopt import hp
search_space = {'lstm_layers': hp.choice('lstm_layers', [1,2, 3]),
                'lstm_filters': hp.quniform('lstm_filters', 8, 16, 1), 
                'lstm_filter_shape': hp.choice('lstm_filter_shape', [[2,2], [3,3], [4,4], [5,5], [6,6]]),
                'lstm_activation': hp.choice('lstm_activation', ['tanh', 'relu']),
                
                'hidden_layers': hp.choice('hidden_layers', [1,2,3]),
                'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1),
                'layers_activation': hp.choice('layers_activation', ['tanh', 'relu'])}


for sample, sample_name in enumerate([data_names[0]]):
    print('\n'+sample_name)
    x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.1)
    x_test, y_test = dataset_loader.get_test_sample(sample)
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  
            
    save_path = os.path.join(results_dir, sample_name)
    functions.save_folder(save_path)   

    model = OptimizableModel(x_train, y_train, x_valid, y_valid, x_test, y_test,
                             model_type = model_name)       
    best, trials = model.optimize(search_space, iters=100)
    print(best)