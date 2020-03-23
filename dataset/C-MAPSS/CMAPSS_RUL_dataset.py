import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import utilities.data_processing as dp

plt.close('all')


def dataset_sampling(dataset_dict, time_window):
     
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['rul_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['rul_test']
    x_train_op_settings = dataset_dict['x_train_op_settings']
    x_test_op_settings = dataset_dict['x_test_op_settings']
    
    h_train = dataset_dict['y_train']
    h_test = dataset_dict['y_test']
    
    
    for i in range(len(x_train)):
        x_train[i] = np.column_stack((x_train[i], x_train_op_settings[i]))
        
    for i in range(len(x_test)):
        x_test[i] = np.column_stack((x_test[i], x_test_op_settings[i]))


    #Health sampling    
    x_train, y_train = dp.health_prediction_sampling_from_truth(x_train, y_train, h_train, time_window,only_remove_samples_wihtout_deg=True)    
    x_test, y_test = dp.health_prediction_sampling_from_truth(x_test, y_test, h_test, time_window, only_remove_samples_wihtout_deg=True)    
 
    #Operation time add
    x_train, y_train = dp.op_time_add(x_train, y_train)
    x_test, y_test = dp.op_time_add(x_test, y_test)

    #Time window sampling 
    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)

    dataset = {'x_train': x_train, 
               'y_train': y_train,
               'x_test': x_test, 
               'y_test': y_test}
    return dataset


def create_npz():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_raw_npz = dict(np.load('CMAPSS_dataset_h.npz', allow_pickle=True))['dataset'][()]
    
    data_dict = {}
    for dataset in datasets: 
        print(dataset)

        time_window = 10
        data_dict[dataset] = dataset_sampling(dataset_raw_npz[dataset], time_window)
    
    np.savez('CMAPSS_dataset.npz',
             dataset = data_dict)
    return None
create_npz()