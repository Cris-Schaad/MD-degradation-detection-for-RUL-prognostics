import os
from models.ANNModel import ANNModel
from utils.dataset_importer import CMAPSS_importer

from utils.training_functions import plt_close
from utils.training_functions import MinMaxScaler
from utils.training_functions import ResultsSaver
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model_name = 'CNN'

results_dir =  os.path.join(os.getcwd(), 'training_results', dataset_name, model_name)
dataset_loader = CMAPSS_importer(dataset_name)
plt_close()


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
        Model.model_save(model, os.path.join(saver.save_folder_dir,'model_'+str(i+1)))
        
        test_loss = rmse_eval(y_test, y_pred, sub_dataset)
        saver.save_iter(i+1, test_loss)
Model.model_plot(model, results_dir, model_name=dataset_name+'_model.png')