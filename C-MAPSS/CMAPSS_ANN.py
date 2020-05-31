import os
import sys
sys.path.append('..')

from ANNModel import ANNModel
from CMAPSS_utils import CMAPSS_importer
from CMAPSS_utils import close_all
from utils.data_processing import MinMaxScaler
from utils.training_functions import ResultsSaver
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model_name = 'CNN'

results_dir =  os.path.join(os.getcwd(), 'training_results', dataset_name, model_name)
dataset_loader = CMAPSS_importer(dataset_name)
close_all()


params = {'cnn_layers': 4,
        'cnn_filters': 168, 
        'cnn_activation': 'relu',
        'cnn_kernel_height': 1,
        'cnn_kernel_width':  5,
        'cnn_padding': 'same',
      
        'hidden_layers': 2, 
        'layers_neurons': 200, 
        'layers_activation': 'relu',
        
        'dropout': 0.25,
        'LR': 0.001,
        'LR_patience': 5, 
        'ES_patience': 10}


for sub_dataset in dataset_loader.subdatasets:
    print('\n'+sub_dataset)

    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.15)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)

    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  
      
    saver = ResultsSaver(results_dir, sub_dataset, sub_dataset)
    Model = ANNModel(x_train, y_train, x_valid, y_valid, x_test, y_test, model_type=model_name)          
    
    for i in range(10):
        model = Model.model_train(params)
        y_pred = model.predict(x_test)
        
        prediction_plots(y_test, y_pred, plot_name=sub_dataset)
        test_loss = rmse_eval(y_test, y_pred, sub_dataset)
        saver.save_iter(i+1, test_loss)
        
        # Test set results saving
        x_test, y_test = dataset_loader.get_samples('test', sub_dataset)
        y_pred = []
        for x_sample in x_test:
            y_pred.append(model.predict(scaler_x.transform(x_sample)))
        np.savez(os.path.join(results_dir, 'model_'+str(i+1)+'_results.npz'), 
                 y_true = y_test,
                 y_pred = y_pred) 
        
        if dataset_name == 'CMAPSS_unfiltered':
            x_test, y_test = dataset_loader.get_samples('test_ignored', sub_dataset)
            y_pred = []
            for x_sample in x_test:
                y_pred.append(model.predict(scaler_x.transform(x_sample)))
            np.savez(os.path.join(results_dir, 'model_'+str(i+1)+'ignored_samples_results.npz'), 
                     y_true = y_test,
                     y_pred = y_pred) 
Model.model_plot(model, results_dir, model_name=dataset_name+'_model.png')