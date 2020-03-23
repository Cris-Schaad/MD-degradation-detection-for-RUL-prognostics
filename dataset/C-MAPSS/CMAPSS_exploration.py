import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import utilities.data_processing as dp

plt.close('all')



def windows(x, time_window, time_step):
    wins = []
    indx = len(x)
    indx_left = True
    while indx_left:
        sample = x[indx-time_window:indx]
        wins.append(sample[::-1])
        indx = indx-time_step
        if indx-time_window < 0:
            indx_left=False
    return np.asarray(wins[::-1])


def plot_vars(healthy_data, middle_data, fault_data, var_1=0, var_2=1):
    
    plt.figure()
    plt.scatter(middle_data[:,var_1], middle_data[:,var_2], 
               c='yellow', alpha=0.5, label='Middle')
    plt.scatter(healthy_data[:,var_1], healthy_data[:,var_2], 
               c='green', alpha=0.5, label='Cluster')
    plt.scatter(fault_data[:,var_1], fault_data[:,var_2], 
               c='red', alpha=1, label='Fault')
    

def dataset_exploration(dataset_dict, dataset):
 
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']
#    x_train_op_settings = dataset_dict['x_train_op_settings']
#    x_test_op_settings = dataset_dict['x_test_op_settings']
    
#    x_train_tot = []
#    for i in range(len(x_train)):
#        x_train_tot.append(np.column_stack((x_train[i], x_train_op_settings[i])))
#    x_train = np.asarray(x_train_tot)
#    
    
#    healthy_cycles = 50
#    healthy_train = []
#    rest_train = []
#    for i in x_train:
#        healthy_train.append(i[:healthy_cycles])
#        rest_train.append(i[healthy_cycles:])
#    healthy_train = np.asarray(healthy_train); rest_train = np.asarray(rest_train)
#    
#    fault_train = []
#    fault_cycles = 10
#    for i in range(len(rest_train)):
#        fault_train.append(rest_train[i][-fault_cycles:])
#        rest_train[i] = rest_train[i][:-fault_cycles]
#    fault_train = np.asarray(fault_train)

#    scaler = MinMaxScaler(feature_range=(0,1))
#    healthy_train = scaler.fit_transform(np.concatenate(healthy_train))  
#    rest_train = scaler.transform(np.concatenate(rest_train))  
#    fault_train = scaler.transform(np.concatenate(fault_train)) 
#    
#    fault_train = scaler.fit_transform(np.concatenate(fault_train))  
#    rest_train = scaler.transform(np.concatenate(rest_train))  
#    healthy_train = scaler.transform(np.concatenate(healthy_train)) 


#    pca = PCA(n_components=14)    
#    healthy_train_pca = pca.fit_transform(healthy_train)
#    rest_train_pca = pca.transform(rest_train)
#    fault_train_pca = pca.transform(fault_train)
#    fault_train_pca = pca.fit_transform(fault_train)
#    rest_train_pca = pca.transform(rest_train)
#    healthy_train_pca = pca.transform(healthy_train)
 

#    plot_vars(healthy_train_pca, rest_train_pca, fault_train_pca, var_1=0, var_2=1)
#    plot_vars(healthy_train_pca, rest_train_pca, fault_train_pca, var_1=1, var_2=2)

    pca = PCA(n_components=14)     
    data_scaler = MinMaxScaler(feature_range=(0,1))
    pca_scaler = MinMaxScaler(feature_range=(0,1))
    pca_scaler.fit_transform(pca.fit_transform(data_scaler.fit_transform(np.concatenate(x_train))))

    
    x_train_filtered = []
    y_train_filtered = []
    
    var_threshold = 0.5
    time_window = 30
    
    if dataset == 'FD001':
        for var in range(14):
            
    #        plt.figure(figsize=(12,4))
            for sample in range(len(x_train)):
                x_sample = pca_scaler.transform(pca.transform(data_scaler.transform(x_train[sample])))
    #            y = x_sample[:,var]
    #            x = np.linspace(1, len(y), len(y))
    #            plt.scatter(x, y, linewidth=0.1, c='C0', s=1)
    #            plt.scatter(x[-1], y[-1], c='r')
                
                if var == 0:                                
                    ind = np.argwhere(x_sample[:,var] > var_threshold)[0][0]
                    x_train_filtered.append(x_train[sample][ind-time_window:])
                    y_train_filtered.append(y_train[sample][ind-time_window:])   
                
                    
        x_test_filtered = []
        y_test_filtered = []     
            
        left_out = []
        for sample in range(len(x_test)):
            x_sample = pca_scaler.transform(pca.transform(data_scaler.transform(x_test[sample])))   
    
            ind = np.argwhere(x_sample[:,0] > var_threshold)
            if len(ind)>0:
                x_test_filtered.append(x_test[sample][ind[0][0]-time_window:])
                y_test_filtered.append(y_test[sample][ind[0][0]-time_window:])  
            else:
                left_out.append(y_test[sample][-1])           
        
        x_train = np.asarray(x_train_filtered)
        y_train = np.asarray(y_train_filtered)
        
        x_test = np.asarray(x_test_filtered)
        y_test = np.asarray(y_test_filtered)    
 
    print(y_test.shape)
    #Time window sampling 
    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)

    dataset = {'x_train': x_train, 
               'y_train': y_train,
               'x_test': x_test, 
               'y_test': y_test}
    return dataset


        


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_raw_npz = dict(np.load('CMAPSS_raw.npz', allow_pickle=True))


data_dict = {}
for i, dataset in enumerate(datasets): 
    print('\n'+dataset)
    data_dict[dataset] = dataset_exploration(dataset_raw_npz[dataset][()], dataset)

np.savez('CMAPSS_dataset.npz',
         dataset = data_dict)