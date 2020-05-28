import os
from models.OptimizableModel import OptimizableModel
from utils.dataset_importer import CMAPSS_importer
from utils.training_functions import plt_close
from utils.training_functions import MinMaxScaler
from utils.training_functions import save_folder
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


model_name = 'CNN'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'C-MAPSS', model_name)
dataset_loader = CMAPSS_importer('CMAPSS_dataset.npz')
plt_close()


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
        'LR_patience':5, 
        'ES_patience': 10}


for sub_dataset in dataset_loader.subdatasets:
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.25)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)

    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  
    
    save_path = os.path.join(results_dir, sub_dataset)
    save_folder(save_path)    
    Model = OptimizableModel(x_train, y_train, x_valid, y_valid, x_test, y_test, model_type=model_name)          
    for i in range(10):
        
        model = Model.model_train(params, plot_model_path=save_path)
        y_pred = model.predict(x_test)

        prediction_plots(y_test, y_pred, plot_name=sub_dataset)
        test_loss = rmse_eval(y_test, y_pred, sub_dataset)
        Model.model_save(model, save_path+'{:.2f}'.format(test_loss))