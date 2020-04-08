import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def mahalanobis_distance(x):
    
    n_vars = x.shape[1]
    z = np.ones_like(x)
    for i in range(n_vars):
        z[:,i] = (x[:,i]-np.mean(x[:,i]))/np.std(x[:,i])
    
    
    md = np.zeros_like(x)
    cov_matrix = np.matmul(np.transpose(z), z)

    for t in range(len(z)):
        z_t = np.expand_dims(z[t,:], axis=1)
        
        MD = np.matmul(np.linalg.inv(cov_matrix), z_t)/n_vars
        md[t] = np.squeeze(np.matmul(np.transpose(z_t), MD), axis=-1)
    return md


def sample_MD_plot(dataset_dict, dataset):
 
    x_train = dataset_dict['x_train']

    plt.figure()
    for sample in x_train[25:35]:
        print(mahalanobis_distance(sample).shape)
        plt.plot(mahalanobis_distance(sample))


plt.close('all')

datasets = ['FD001', 'FD002', 'FD003', 'FD004']
dataset_raw_npz = dict(np.load('CMAPSS_raw.npz', allow_pickle=True))

for i, dataset in enumerate(['FD003']): 
    print('\n'+dataset)
    sample_MD_plot(dataset_raw_npz[dataset][()], dataset)