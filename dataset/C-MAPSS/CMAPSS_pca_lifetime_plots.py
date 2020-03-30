import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



plt.close('all')

  

def pca_lifetime_data_plot(dataset_dict, dataset):
 
    x_train = dataset_dict['x_train']
    n_vars = x_train[0].shape[1]

    pca = PCA(n_components=n_vars)     
    data_scaler = MinMaxScaler(feature_range=(0,1))
    pca_scaler = MinMaxScaler(feature_range=(0,1))
    pca_scaler.fit_transform(pca.fit_transform(data_scaler.fit_transform(np.concatenate(x_train))))

    for var in range(x_train[0].shape[1]):
        
        plt.figure(figsize=(12,4))
        for sample in range(len(x_train)):
            x_sample = pca_scaler.transform(pca.transform(data_scaler.transform(x_train[sample])))
           
            y = pca.explained_variance_ratio_[var]*x_sample[:,var]
            x = np.linspace(1, len(y), len(y))
            plt.scatter(x, y, linewidth=0.1, c='C0', s=1)
            plt.scatter(x[-1], y[-1], c='r')


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_raw_npz = dict(np.load('CMAPSS_raw.npz', allow_pickle=True))

for i, dataset in enumerate(['FD003']): 
    print('\n'+dataset)
    pca_lifetime_data_plot(dataset_raw_npz[dataset][()], dataset)
