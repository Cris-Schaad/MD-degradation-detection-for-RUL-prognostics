import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


import MahalanobisDistance as MD
import degradation_detection as dd
import data_processing as dp


plt.close('all')


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
data_dir = os.path.join('C-MAPSS', 'processed_data')
dataset_raw_npz = dict(np.load(os.path.join(data_dir,'CMAPSS_raw.npz'), allow_pickle=True))


time_window = 30
healthy_cycles = 10

    
data_dict = {}
for i, dataset in enumerate(datasets): 
    
    print('Processing dataset: ', dataset)
    dataset_dict = dataset_raw_npz[dataset][()]
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']

    x_train_healthy = np.asarray([i[:healthy_cycles] for i in x_train])
    x_train_healthy = np.asarray([i for i in x_train_healthy])
#    x_train_rest = np.asarray([x_scaler.transform(i) for i in x_train_rest])
    
    m_d = MD.MahalanobisDistance(mode='inv')
    m_d.fit_predict(np.concatenate(x_train_healthy))
    
    healthy_md = np.asarray([m_d.fit(i) for i in x_train])
        
    plt.figure()
    ini_md = []; end_md = []
    i_max = 0; min_dif = np.inf
    for i in healthy_md:
        if dd.cum_mean(i)[-1] < min_dif:
            i_max = i; min_dif = dd.cum_mean(i)[-1] 
        plt.plot(dd.cum_mean(i))
        ini_md.append(np.diff(dd.cum_mean(i)[:100]))
        end_md.append(np.diff(dd.cum_mean(i)[-10:]))
        
    plt.title(dataset + ' training set')
    plt.xlabel('Flight cycles')
    plt.ylabel('Mahalanobis distance (cummulativa average)')

#    plt.figure()
#    p, bins, patches = plt.hist(np.asarray(ini_md).flatten(), range=(-0.1, 0.1), bins=500, density=True)
#    plt.hist(np.asarray(end_md).flatten(), bins=500, density=True)
    
#    x = (bins[:-1]+bins[1:])/2
#    mean = dist_expected_value(x, p)
#    plt.axvline(mean, c='r')
#    
#    print(min_dif)
#    indx = detector(cum_mean(i_max), 0, k=8, n_con=10)
#    plt.figure()
#    plt.plot(cum_mean(i_max))
#    plt.axhline(mean, c='r')
#    plt.axvline(range(len(cum_mean(i_max)))[indx])
#    
    i_min = 0
    for i in healthy_md:
        ind = len(dd.cum_mean(i)) - dd.detector(dd.cum_mean(i), 0, k=12, n_con=15)
        if ind > i_min:
            i_min = ind; imin=i
            
    plt.figure()
    plt.plot(dd.cum_mean(imin))
    plt.axvline(range(len(dd.cum_mean(imin)))[dd.detector(dd.cum_mean(imin), 0, k=12, n_con=15)], c='r')
    
    #Time window sampling 
    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)

    data_dict[dataset] =  {'x_train': x_train,
                         'y_train':y_train,
                         'x_test': x_test, 
                         'y_test': y_test}

np.savez(os.path.join(data_dir,'CMAPSS_dataset.npz'),
         dataset = data_dict)