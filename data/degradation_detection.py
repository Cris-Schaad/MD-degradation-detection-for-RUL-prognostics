import numpy as np


class Detector():
    
    def __init__(self, k, n, moving_window=1):
        self.k = k
        self.n = n
        self.window = moving_window

    def detect(self, x, threshold, verbose=False):
        
        x = self.moving_average(x)
        for i in range(self.n-1, len(x)):
            sample_k = 0
            for m in range(self.n): 
                if x[i - (self.n-1) + m] >= threshold:
                    sample_k = sample_k +1
                
            if sample_k >= self.k:
                return int(i + self.window)
            
            if i == len(x)-1:
                if verbose:
                    print('No tresspassing detected')
                return int(i + self.window)
            
    
    def sampling_from_index(self, x, y, ind, time_window):
        
        x_sampled = []
        y_sampled = []
        for x_sample, y_sample, deg_start_index in zip(x, y, ind):
            if deg_start_index < len(x_sample)-1:
                ind_start = deg_start_index-time_window if deg_start_index-time_window > 0 else 0
                x_sampled.append(x_sample[ind_start:])
                y_sampled.append(y_sample[ind_start:]) 
        return np.asarray(x_sampled), np.asarray(y_sampled)
    
    
    def moving_average(self, x):
        if self.window > 1:
            return np.asarray([np.mean(x[i-(self.window):i]) for i in range(self.window, len(x))])
        else:
            return x
