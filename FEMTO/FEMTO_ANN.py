import os
import sys
import numpy as np

sys.path.append('..')
from ANNModel import ANNModel
from FEMTO_utils import FEMTO_importer
from utils.data_processing import MinMaxScaler
from utils.training_functions import close_all
from utils.training_functions import ResultsSaver
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


model_name = 'ConvLSTM'
dataset_name = 'FEMTO'

results_dir =  os.path.join(os.getcwd(), 'training_results', dataset_name, model_name)
dataset_loader = FEMTO_importer(dataset_name)
close_all()

params = {'convlstm_layers': 1,
        'convlstm_filters': 22, 
        'convlstm_activation': 'relu',
        'convlstm_kernel_height': 8,
        'convlstm_kernel_width': 8,
      
        'hidden_layers': 4, 
        'layers_neurons': 208, 
        'layers_activation': 'relu',
        
        'dropout': 0.5,
        'LR': 0.001,
        'LR_patience': 5, 
        'ES_patience': 10}


for sample, sample_name in enumerate(dataset_loader.data_names):
    print('\n'+sample_name)
    
    x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.2)
    x_test, y_test = dataset_loader.get_test_sample(sample)
    
    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)

            
    saver = ResultsSaver(results_dir, sample_name, sample_name)
    Model = ANNModel(x_train, y_train, x_valid, y_valid, x_test=x_test, y_test=y_test,
                     model_type=model_name)      
         
    for i in range(10):
        model = Model.model_train(params)
        y_pred = model.predict(x_test)

        prediction_plots(y_test, y_pred, plot_name=sample_name+'_iter_'+str(i+1), save_dir=os.path.join(results_dir, sample_name))
        test_loss = rmse_eval(y_test, y_pred, sample_name)
        saver.save_iter(i+1, test_loss)    
        
        # Test set results saving
        np.savez(os.path.join(results_dir, sample_name,'model_'+str(i+1)+'_results.npz'), 
                 y_true = y_test,
                 y_pred = y_pred) 

Model.model_plot(model, results_dir, model_name=dataset_name+'_model.png')