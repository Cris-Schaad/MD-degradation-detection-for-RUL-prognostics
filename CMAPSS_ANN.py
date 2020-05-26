import os
import time
import numpy as np


from models.OptimizableModel import OptimizableModel
from utils import dataset_importer
import utils.training_functions as functions


model_name = 'CNN'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'C-MAPSS', model_name)
dataset_dir = os.path.join(os.getcwd(), 'data', 'C-MAPSS', 'processed_data', 'CMAPSS_dataset.npz')
dataset_npz = dict(np.load(dataset_dir, allow_pickle=True))['dataset'][()]
dataset_loader = dataset_importer.CMAPSS_importer(dataset_dir)
functions.plt_close()


# params = {'EarlyS_patience': 23.0, 'LRswitch_patience': 28.0, 'cnn_activation': 'relu', 'cnn_filters': 67.0, 'cnn_kernel_height': 3.0, 
#           'cnn_kernel_width': 4.0, 'cnn_layers': 3, 'cnn_padding': 'same', 'hidden_layers': 1, 'layers_activation': 'relu',
#           'layers_neurons': 140.0}

params = {'cnn_layers': 4,
            'cnn_filters': 168, 
            'cnn_activation': 'relu',
            'cnn_kernel_height': 1,
            'cnn_kernel_width':  5,
            'cnn_padding': 'same',
          
            'hidden_layers': 2, 
            'layers_neurons':200, 
            'layers_activation': 'relu',
            
            'dropout': 0.25,
            'LR': 0.001, 
            'ES_patience': 1}
            # 'ES_patience': 12}


for sub_dataset in dataset_npz.keys():
    
    if sub_dataset == 'FD001':
        params['LR_patience'] = 4
    
    else:
        params['LR_patience'] = 8
        
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.25)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)

    scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  
    
    iters = 1
    save_path = os.path.join(results_dir, sub_dataset)
    functions.save_folder(save_path)     

    Model = OptimizableModel(x_train, y_train, x_valid, y_valid, x_test, y_test,model_type=model_name)          
    for i in range(iters):
        
        model = Model.model_train(params, plot_model_path=save_path)
        y_pred = model.predict(x_test)

        plot_name = str('C-MAPSS RUL '+sub_dataset+' prediction iter '+str(i+1))
        functions.prediction_plots(y_test, y_pred, plot_name, save_dir=save_path, xlabel='Engne sample', ylabel='RUL')
        functions.print_rmse(y_test, y_pred, sub_dataset)
#        functions.loss_plot(model_history.history['loss'], model_history.history['val_loss'])
        
        # y_pred_test = []
        # x_test_complete, y_test_true = dataset_loader.get_test_samples(sub_dataset)    
        # for sample in x_test_complete:
        #     x_sample = scaler_x.transform(sample)
        #     y_pred_test.append(scaler_y.inverse_transform(model.predict(x_sample)))

        # functions.save_results(y_test_true, np.asarray(y_pred_test), save_path, name='iter_'+str(i+1))