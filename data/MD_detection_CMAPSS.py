import os
import numpy as np
import matplotlib.pyplot as plt

import MS_iterator as MS_iterator
import data_processing as dp


plt.close('all')

data_dir = os.path.join('C-MAPSS', 'processed_data')
dataset_raw_npz = dict(np.load(os.path.join(data_dir,'CMAPSS_raw.npz'), allow_pickle=True))


sigmas = [0.3, 0.35, 0.5, 0.5]
data_dict = {}
md_dict = {}

for i, dataset in enumerate(['FD001', 'FD002', 'FD003', 'FD004']): 
    
    print('Processing dataset: ', dataset)
    dataset_dict = dataset_raw_npz[dataset][()]
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']
        
    #Degradation detector parameters
    k = 5; n = 5
    sigma = sigmas[i]

    iterator = MS_iterator.iterator(k, n, sigma)
    deg_start_ind, threshold = iterator.iterative_calculation(x_train, verbose=False)
    
    iterator.plot_RUL_after_deg(x_train, y_train, plot=False)
    iterator.plot_RUL_after_deg(x_test, y_test, plot=False)
    iterator.plot_lifetime_dist(x_test, y_test)
    
    md_dict[dataset] =  {'x_train_md': iterator.md_calculation_op(x_train),
                        'x_test_md': iterator.md_calculation_op(x_test),
                        'threshold': threshold,
                        'ms_iter': np.asarray(iterator.iter_ms_dim)}
        
    # Sampling from degradation start index
    time_window = 15
    x_train, y_train = iterator.sampling_from_MD_deg_start(x_train, y_train, time_window)
    x_test, y_test = iterator.sampling_from_MD_deg_start(x_test, y_test, time_window)
 
    
    #Time window sampling 
    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)
    

    data_dict[dataset] =  {'x_train': x_train,
                          'y_train':y_train,
                          'x_test': x_test, 
                          'y_test': y_test}

np.savez(os.path.join(data_dir,'CMAPSS_dataset.npz'),
          dataset = data_dict)
np.savez(os.path.join(data_dir,'CMAPSS_md_dataset.npz'),
          dataset = md_dict)