import numpy as np

import MahalanobisDistance as MD
import degradation_detection as dd


class iterator():
    
    def __init__(self, k, n, sigma, initial_deg_start_index, moving_window=None, norm_by_start=True):
        """
        k: number of conditions out of "n" consecutive points that are out of Mahalanobis Space
        n: number of consecutive conditions
        sigma: standard deviation that defines the Mahalanobis Space
        """
        
        self.k = k
        self.n = n
        self.s = sigma
        
        self.norm_by_start = norm_by_start
        self.window = moving_window
        self.initial_deg_start_index = initial_deg_start_index
        
        self.iter_ms_mean = []
        self.iter_ms_std = []
        self.iter_ms_dim = []
        
        
    def iterative_calculation(self, x, n_iterations=10, verbose=True):
        
        
        self.m_d = MD.MahalanobisDistance(mode='covariance')
        self.detector = dd.Detector(self.k, self.n)

        deg_start_indx = [self.initial_deg_start_index for _ in range(len(x))]
        for i in range(n_iterations):
            
            x_healthy = np.asarray([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x)])    
            self.m_d.fit_predict(np.concatenate(x_healthy))
            x_md = self.md_calculation_op(x)     
            
            ms = np.concatenate([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x_md)])
            threshold = np.mean(ms) + self.s*np.std(ms)
    
            if verbose:
                print('Iteration: {:}'.format(int(i+1)))
                print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
                print('\tThreshold: {:.2f}'.format(threshold), '\n')
            
            
            self.iter_ms_mean.append(np.mean(ms))
            self.iter_ms_std.append(np.std(ms))
            self.iter_ms_dim.append(len(ms))
            
            for ind, md_sample in enumerate(x_md):
                deg_start_indx[ind] = self.detector.detect(md_sample, threshold, prints=verbose) + self.window-1
                              
        print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
        print('\tThreshold: {:.2f}'.format(threshold), '\n')       
        
        for ind, md_sample in enumerate(x_md):
            self.detector.detect(md_sample, threshold, prints=True) 
            
        return deg_start_indx, threshold
    
    
    def md_calculation_op(self, x):
        
        if self.window is not None:
            x_md = np.asarray([self.moving_average(self.m_d.fit(sample)) for sample in x])
        else:
            x_md = np.asarray([self.m_d.fit(sample) for sample in x])
            
        if self.norm_by_start:
            x_md = self.relative_to_start(x_md)   
        return x_md
        
    
    def moving_average(self, x):
        return np.asarray([np.mean(x[i-(self.window-1):i]) for i in range(self.window-1, len(x))])
    
    
    def relative_to_start(self, x):
        for i, sample in enumerate(x):
            x[i] = sample/sample[0]
        return x
    
    
    def get_detector(self):
        return self.detector
