import os
import numpy as np
import matplotlib.pyplot as plt

from MSIterativeAlgorithm import Detector
from MSIterativeAlgorithm import MSIterativeAlgorithm

from data_processing import get_CMAPSS_dataset
from data_processing import get_CMAPSS_MD
from data_processing import time_window_sampling
from data_processing import samples_under_deg



plt.close('all')
DATA_DIR = os.path.join('C-MAPSS', 'processed_data')


def MD_calculation_CMAPSS():
    sigmas = [0.3, 0.35, 0.4, 0.5]
    md_dict = {}
    
    for i, dataset in enumerate(['FD001', 'FD002', 'FD003', 'FD004']): 
        print('Processing dataset: ', dataset)
        x_train, y_train, x_test, y_test = get_CMAPSS_dataset(dataset, DATA_DIR)
            
        #Degradation detector parameters
        k = 5; n = 5
        sigma = sigmas[i]
        iterator = MSIterativeAlgorithm(k, n, sigma)
        iterator.iterative_calculation(x_train, verbose=False)
        
        iterator.RUL_info(x_train, y_train, plot=False)
        iterator.RUL_info(x_test, y_test, plot=False)
        iterator.plot_lifetime_dist(x_test, y_test, savepath='C-MAPSS', figname=dataset)
        
        md_dict[dataset] =  {'x_train_md': iterator.md_calculation_op(x_train),
                            'x_test_md': iterator.md_calculation_op(x_test),
                            'threshold': iterator.threshold,
                            'ms_iter': np.asarray(iterator.iter_ms_dim)}
    np.savez(os.path.join(DATA_DIR,'CMAPSS_md_dataset.npz'), **md_dict)
# MD_calculation_CMAPSS()


def sample_CMAPSS_by_MD():
    
    data_filtered = {}
    data_unfiltered = {}
    
    k = 5; n = 5
    deg_detector = Detector(k, n)
    
    for i, dataset in enumerate(['FD001', 'FD002', 'FD003', 'FD004']):        
        print('Processing dataset: ', dataset)
        x_train, y_train, x_test, y_test = get_CMAPSS_dataset(dataset, DATA_DIR)
        x_train_md, x_test_md, threshold = get_CMAPSS_MD(dataset, DATA_DIR)

        # Degradation start index
        time_window = 18
        train_deg = deg_detector.dataset_detect(x_train_md, threshold)
        test_deg = deg_detector.dataset_detect(x_test_md, threshold)
         
        
        # Sampling from degradation start index
        x_train, y_train = deg_detector.sampling_from_index(x_train, y_train, train_deg, time_window)
        x_test, y_test = deg_detector.sampling_from_index(x_test, y_test, test_deg, time_window)
        

        #Time window sampling 
        x_train, y_train = time_window_sampling(x_train, y_train, time_window, add_last_dim=True)
        x_test, y_test = time_window_sampling(x_test, y_test, time_window, add_last_dim=True)
        
        data_filtered[dataset] =  {'x_train': x_train,
                                   'y_train':y_train,
                                   'x_test': x_test, 
                                   'y_test': y_test}
        
        
        #Unfiltered data
        _, _, x_test, y_test = get_CMAPSS_dataset(dataset, DATA_DIR)
        x_test_normal, x_test_under_deg = samples_under_deg(x_test, test_deg)
        y_test_normal, y_test_under_deg = samples_under_deg(y_test, test_deg)
        
        
        #Time window sampling
        x_test_normal, y_test_normal = time_window_sampling(x_test_normal, y_test_normal, time_window, add_last_dim=True)
        x_test_under_deg, y_test_under_deg = time_window_sampling(x_test_under_deg, y_test_under_deg, time_window, add_last_dim=True)
        
        data_unfiltered[dataset] =  {'x_test_ignored': x_test_normal,
                                     'y_test_ignored':y_test_normal,
                                     'x_test': x_test_under_deg,
                                     'y_test': y_test_under_deg}
    np.savez(os.path.join(DATA_DIR,'CMAPSS_dataset.npz'), **data_filtered)
    np.savez(os.path.join(DATA_DIR,'CMAPSS_unfiltered_dataset.npz'), **data_unfiltered)
sample_CMAPSS_by_MD()