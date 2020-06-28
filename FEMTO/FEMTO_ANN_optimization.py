import sys

sys.path.append('..')
from ANNModel import ANNModel
from FEMTO_utils import FEMTO_importer
from utils.data_processing import MinMaxScaler
from utils.training_functions import close_all


model_name = 'ConvLSTM'
dataset_loader = FEMTO_importer('FEMTO')
close_all()


scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=1)
from hyperopt import hp
search_space =  {'convlstm_layers': 1,
                'convlstm_filters': 22, 
                'convlstm_activation': 'relu',
                'convlstm_kernel_height': hp.quniform('convlstm_kernel_height', 2,12,1),
                'convlstm_kernel_width': hp.quniform('convlstm_kernel_width', 2,12,1),
              
                'hidden_layers': hp.quniform('hidden_layers', 1,5,1), 
                'layers_neurons': hp.quniform('layers_neurons', 8,256,8), 
                'layers_activation': 'relu',
                
                'dropout': hp.quniform('dropout', 0, 0.8, 0.001),
                'LR': 0.001,
                'LR_patience': 5, 
                'ES_patience': 10}


sample = 2
sample_name =   dataset_loader.data_names[sample]

print('\n'+sample_name)
x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.33)

x_train = scaler_x.fit_transform(x_train)
x_valid = scaler_x.transform(x_valid)

model = ANNModel(x_train, y_train, x_valid, y_valid, x_test=None, y_test=None, model_type=model_name)       
best, trials = model.optimize(search_space, iters=100)
print(best)