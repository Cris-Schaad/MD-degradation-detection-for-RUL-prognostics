import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt


#import utilities.data_processing as dp
import MahalanobisDistance as MD

plt.close('all')


def dataset_exploration(dataset_dict, dataset):
 
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']


    var_threshold = 0.5
    time_window = 30
    
    healthy_cycles = 10
    x_train_healthy = np.asarray([i[:healthy_cycles] for i in x_train])
    
    x_train_healthy = np.asarray([i for i in x_train_healthy])
#    x_train_rest = np.asarray([x_scaler.transform(i) for i in x_train_rest])
    
    m_d = MD.MahalanobisDistance(mode='inv')
    m_d.fit_predict(np.concatenate(x_train_healthy))
    
    healthy_md = np.asarray([m_d.fit(i) for i in x_train])
    
    plt.figure()
    ini_md = []
    end_md = []
    for i in healthy_md:
        plt.plot(i)
        ini_md.append(i[:10].flatten())
        end_md.append(i[-10:])
    

    plt.figure()
    plt.hist(np.concatenate(ini_md).flatten(), bins=30)
    plt.hist(np.concatenate(end_md).flatten(), bins=900)
#    
#    print(y_test.shape)
#    #Time window sampling 
#    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
#    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)

    dataset = {'x_train': x_train, 
               'y_train': y_train,
               'x_test': x_test, 
               'y_test': y_test}
    return dataset


        


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_raw_npz = dict(np.load('CMAPSS_raw.npz', allow_pickle=True))


data_dict = {}
for i, dataset in enumerate(['FD001']): #enumerate(datasets): 
    print('\n'+dataset)
    data_dict[dataset] = dataset_exploration(dataset_raw_npz[dataset][()], dataset)

np.savez('CMAPSS_dataset.npz',
         dataset = data_dict)