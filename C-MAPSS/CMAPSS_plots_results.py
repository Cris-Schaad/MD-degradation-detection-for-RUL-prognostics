import os
import sys
sys.path.append("..")

import numpy as np
from tensorflow.keras.models import load_model

from CMAPSS_utils import close_all
from CMAPSS_utils import CMAPSS_importer
from utils.data_processing import MinMaxScaler
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model = 'CNN'

models_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)
dataset_loader = CMAPSS_importer(dataset_name)
close_all()


for sub_dataset in dataset_loader.subdatasets:
    saved_models_list = [i for i in os.listdir(os.path.join(models_dir, sub_dataset)) if '.' not in i]
    loaded_model = load_model(os.path.join(models_dir, sub_dataset, saved_models_list[0]))

    x_train, _, _, _ =  dataset_loader.get_train_set(sub_dataset, valid_size=0.15)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)


    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)  
    
    y_pred = loaded_model.predict(x_test)
    prediction_plots(y_test, y_pred, plot_name=sub_dataset)    
    test_loss = rmse_eval(y_test, y_pred, sub_dataset)
    
    
    # test_lens = [len(x) for i, x in enumerate(x_test)]
    # min_len_sample_ind = np.argwhere(test_lens == np.min(test_lens))[0][0]
    # max_len_sample_ind = np.argwhere(test_lens == np.max(test_lens))[0][0]
    
    # x_sample = x_test[max_len_sample_ind]
    # y_sample = y_test[max_len_sample_ind]