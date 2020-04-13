import os 
import numpy as np


class Detector():
    
    def __init__(self, k, n, threshold):
        self.k = k
        self.n = n
        self.threshold = threshold

    def detect(self, x, prints=False):
        for i in range(self.n, len(x)):
            sample_k = 0
            for m in range(self.n): 
                if x[i - self.n + m +1] >= self.threshold:
                    sample_k = sample_k +1
                
            if sample_k >= self.k:
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
            
            if ind[i] < len(x_sample)-1:
                if ind[i] - time_window > 0:
                    x_sampled.append(x_sample[ind[i]-time_window:])
                    y_sampled.append(y_sample[ind[i]-time_window:])
                if ind[i] - time_window <= 0:
                    x_sampled.append(x_sample[0:])
                    y_sampled.append(y_sample[0:])   
                    
        return np.asarray(x_sampled), np.asarray(y_sampled)
