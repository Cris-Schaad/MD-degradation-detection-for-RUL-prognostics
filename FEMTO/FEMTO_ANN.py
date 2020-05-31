import os
import time
import numpy as np


from models.OptimizableModel import OptimizableModel
from utils import dataset_importer
import utils.training_functions as functions


model_name = 'CNN'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'FEMTO', model_name)
dataset_dir = os.path.join(os.getcwd(), 'data', 'FEMTO', 'processed_data', 'FEMTO_dataset.npz')
dataset_loader = dataset_importer.FEMTO_importer(dataset_dir)
functions.plt_close()

data_names = dataset_loader.data_names
scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=2)
scaler_y = functions.MinMaxScaler(feature_range=(0,1))

params = {'hidden_layers': 2, 'layers_activation': 'relu', 'layers_neurons': 212.0, 'lstm_activation': 'relu', 
          'lstm_filter_shape': (5, 5), 'lstm_filters': 20.0, 'lstm_layers': 2}


for sample, sample_name in enumerate(data_names):
    print('\n'+sample_name)
    x_train, x_valid, y_train, y_valid = dataset_loader.get_train_set(sample, valid_size=0.1)
    x_test, y_test = dataset_loader.get_test_sample(sample)
    
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)
        
    y_train = scaler_y.fit_transform(y_train)
    y_valid = scaler_y.transform(y_valid)
    y_test = scaler_y.transform(y_test)
            
    iters = 1
    save_path = os.path.join(results_dir, sample_name)
    functions.save_folder(save_path)   
    
    Model = OptimizableModel(x_train, y_train, x_valid, y_valid, x_test, y_test,
                             model_type=model_name)      
         
    for i in range(iters):
        model = Model.model_train(params)
        # print('Time to train {:.2f}'.format(time.time()-start_time))       

        y_pred = model.predict(x_test)
        y_train = scaler_y.inverse_transform(y_train)
        y_valid = scaler_y.inverse_transform(y_valid)
        y_test = scaler_y.inverse_transform(y_test)
        y_pred = scaler_y.inverse_transform(y_pred)

        plot_name = str('FEMTO RUL '+sample_name+' prediction iter '+str(i+1))
        functions.prediction_plots(y_test, y_pred, plot_name, save_dir=save_path, xlabel='Operating time', ylabel='RUL')
        functions.print_rmse(y_test, y_pred, sample_name)
#        functions.loss_plot(model_history.history['loss'], model_history.history['val_loss'])
        