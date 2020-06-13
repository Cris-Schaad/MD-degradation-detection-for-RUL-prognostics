import sys

sys.path.append('..')
from ANNModel import ANNModel
from FEMTO_utils import FEMTO_importer
from utils.data_processing import MinMaxScaler
from utils.training_functions import close_all


model_name = 'CNN'
dataset_loader = FEMTO_importer('FEMTO')
close_all()


scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=1)
from hyperopt import hp
# search_space =  {'convlstm_layers': 2,
#                 'convlstm_filters': 16, 
#                 'convlstm_activation': 'relu',
#                 'convlstm_kernel_height': hp.quniform('convlstm_kernel_height', 4, 12,1),
#                 'convlstm_kernel_width': hp.quniform('convlstm_kernel_width', 4, 12,1),
              
#                 'hidden_layers': 2, 
#                 'layers_neurons': 256, 
#                 'layers_activation': 'relu',
                
#                 'dropout': 0.1,
#                 'LR': 0.001,
#                 'LR_patience': 20, 
#                 'ES_patience': 25}

search_space = {'cnn_layers': hp.quniform('cnn_layers', 1, 5, 1),
                'cnn_filters': hp.quniform('cnn_filters', 8, 256, 16), 
                'cnn_activation': 'relu',
                'cnn_kernel_height': 8,
                'cnn_kernel_width':  8,
                'cnn_padding': 'same',
              
                'hidden_layers': 2, 
                'layers_neurons': 200, 
                'layers_activation': 'relu',
                
                'dropout': 0.25,
                'LR': 0.001,
                'LR_patience': 40, 
                'ES_patience': 50}

sample = 0
sample_name =   dataset_loader.data_names[sample]

print('\n'+sample_name)
x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.33)
x_test, y_test = dataset_loader.get_test_sample(sample)

x_train = scaler_x.fit_transform(x_train)
x_valid = scaler_x.transform(x_valid)
x_test = scaler_x.transform(x_test)  
        

model = ANNModel(x_train, y_train, x_valid, y_valid, x_test=x_test, y_test=y_test, model_type=model_name)       
best, trials = model.optimize(search_space, iters=100)
print(best)