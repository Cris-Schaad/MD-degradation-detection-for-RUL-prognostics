import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA




def plot_vars(healthy_data, middle_data, fault_data, var_1=0, var_2=1):
    
    plt.figure()
    plt.scatter(middle_data[:,var_1], middle_data[:,var_2], 
               c='yellow', alpha=0.5, label='Middle')
    plt.scatter(healthy_data[:,var_1], healthy_data[:,var_2], 
               c='green', alpha=0.5, label='Cluster')
    plt.scatter(fault_data[:,var_1], fault_data[:,var_2], 
               c='red', alpha=1, label='Fault')
    
    
    
def pca_op_plot(dataset_dict):
    x_train = dataset_dict['x_train']
    
    healthy_cycles = 10
    healthy_train = []
    rest_train = []
    for i in x_train:
        healthy_train.append(i[:healthy_cycles])
        rest_train.append(i[healthy_cycles:])
    healthy_train = np.asarray(healthy_train); rest_train = np.asarray(rest_train)
    
    fault_train = []
    fault_cycles = 10
    for i in range(len(rest_train)):
        fault_train.append(rest_train[i][-fault_cycles:])
        rest_train[i] = rest_train[i][:-fault_cycles]
    fault_train = np.asarray(fault_train)

    scaler = MinMaxScaler(feature_range=(0,1))
    healthy_train = scaler.fit_transform(np.concatenate(healthy_train))  
    rest_train = scaler.transform(np.concatenate(rest_train))  
    fault_train = scaler.transform(np.concatenate(fault_train)) 
    
    pca = PCA(n_components=x_train[0].shape[1])    
    healthy_train_pca = pca.fit_transform(healthy_train)
    rest_train_pca = pca.transform(rest_train)
    fault_train_pca = pca.transform(fault_train)

    plot_vars(healthy_train_pca, rest_train_pca, fault_train_pca, var_1=0, var_2=1)
    plot_vars(healthy_train_pca, rest_train_pca, fault_train_pca, var_1=1, var_2=2)
    
    
    
datasets = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_raw_npz = dict(np.load('CMAPSS_raw.npz', allow_pickle=True))

for dataset in datasets:
    print('\n'+dataset)
    pca_op_plot(dataset_raw_npz[dataset][()])