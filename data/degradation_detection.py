import os 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def cum_mean(x, time_window=10):
    return np.asarray([np.mean(x[:i])/np.mean(x[:time_window]) for i in range(time_window, len(x))])



def dist_expected_value(x, p):
    return np.sum(np.dot(np.diff(x),x[1:]*p[1:] + x[:-1]*p[:-1]))/2



def dist_variance(x, p, u):
    return np.sum([p[i]*(x[i] - u)**2 for i in range(len(x))])
    


def detector(x, threshold, k, n_con=10):
    
    for i in range(n_con, len(x)):
        if x[i] >= threshold:
            interval = x[i-n_con:i]
            positive_diff = np.argwhere(np.diff(interval) > 0).flatten()
            if len(positive_diff) >= k-1:
                break        
        if i == len(x) - 1:
            print('No tresspassing detected')
    return i


def health_index(x_true, y_true, degradation_start_indexes):
    
    x = []
    h = []
    y = []
    for i, sample in enumerate(x_true):
        end_rul = y_true[i][-1][0]
        deg_start_index = degradation_start_indexes[i]
        
        if not np.isnan(deg_start_index):
            timesteps_under_deg = len(sample) - int(deg_start_index)
            
            health_without_degradation = np.ones(int(deg_start_index))
            health_under_degradation = np.asarray([(end_rul + i)/(end_rul+timesteps_under_deg) for i in range(timesteps_under_deg)])
            
            health = np.concatenate((health_without_degradation, health_under_degradation[::-1]))
            x.append(x_true[i])
            h.append(np.expand_dims(health, axis=-1))   
            y.append(y_true[i])
    return np.asarray(x), np.asarray(h), np.asarray(y)



def degradation_start_detection(labels, n_conditions, conditions_time_window):
    condition_start_list = []
    
    not_degraded_samples = []
    for j, sample_labels in enumerate(labels):
        
        condition_start_index = 0    
        for i in range(len(sample_labels) - conditions_time_window):
            in_window_labels = sample_labels[i:i+conditions_time_window]
            condition_start_index = i + conditions_time_window
            
            if sum(in_window_labels) >= n_conditions:
                break
    
            if condition_start_index == len(sample_labels) - 1:
                not_degraded_samples.append(j)

        condition_start_list.append(condition_start_index)
    
    if len(not_degraded_samples) > 0:
        print(str(len(not_degraded_samples)) + ' samples without degradation: ', [i+1 for i in not_degraded_samples])
            
    return np.asarray(condition_start_list), np.asarray(not_degraded_samples)

        