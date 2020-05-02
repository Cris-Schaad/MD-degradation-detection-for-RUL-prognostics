import os 
import numpy as np


class Detector():
    
    def __init__(self, k, n):
        self.k = k
        self.n = n

    def detect(self, x, threshold, prints=False):
        for i in range(self.n-1, len(x)):
            sample_k = 0
            for m in range(self.n): 
                if x[i - (self.n-1) + m] <= threshold:
                    sample_k = sample_k +1
                
            if sample_k <= self.k:
                return i
            
            if i == len(x)-1:
                if prints:
                    print('No tresspassing detected')
                return i
            
    
    def sampling_from_index(self, x, y, ind, time_window):
        
        x_sampled=[]; y_sampled=[]
        for i in range(len(x)):
            x_sample = x[i]
            y_sample = y[i]
            deg_stat_index = int(ind[i])
            
            if deg_stat_index < len(x_sample)-1:
                if deg_stat_index - time_window > 0:
                    x_sampled.append(x_sample[deg_stat_index-time_window:])
                    y_sampled.append(y_sample[deg_stat_index-time_window:])
                if deg_stat_index - time_window <= 0:
                    x_sampled.append(x_sample[0:])
                    y_sampled.append(y_sample[0:])   
        
        return np.asarray(x_sampled), np.asarray(y_sampled)
