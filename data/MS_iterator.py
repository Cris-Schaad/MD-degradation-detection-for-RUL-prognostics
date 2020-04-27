import numpy as np

import MahalanobisDistance as MD
import degradation_detection as dd


class iterator():
    
    def __init__(self, k, n, sigma, initial_deg_start_index, start_period):
        """
        k: number of conditions out of "n" consecutive points that are out of Mahalanobis Space
        n: number of consecutive conditions
        sigma: standard deviation that defines the Mahalanobis Space
        """
        
        self.k = k
        self.n = n
        self.s = sigma
        self.start_period = start_period
        self.initial_deg_start_index = initial_deg_start_index
        
        self.iter_ms_mean = []
        self.iter_ms_std = []
        self.iter_ms_dim = []
        
        
    def iterative_calculation(self, x, n_iterations=10, verbose=True):
        
        deg_start_indx = [self.initial_deg_start_index for _ in range(len(x))]
        for i in range(n_iterations):
            
            x_healthy = np.asarray([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x)])    
            m_d = MD.MahalanobisDistance(mode='covariance')
            m_d.fit_predict(np.concatenate(x_healthy))
            
            x_md = np.asarray([self.relative_to_start_period(m_d.fit(sample)) for sample in x])
            
            ms = np.concatenate([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x_md)])
            threshold = np.mean(ms) + self.s*np.std(ms)
    
            if verbose:
                print('Iteration: {:}'.format(int(i+1)))
                print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
                print('\tThreshold: {:.2f}'.format(threshold), '\n')
            
            
            self.iter_ms_mean.append(np.mean(ms))
            self.iter_ms_std.append(np.std(ms))
            self.iter_ms_dim.append(len(ms))
            
            detector = dd.Detector(self.k, self.n, threshold = threshold)
            for ind, md_sample in enumerate(x_md):
                deg_start_indx[ind] = detector.detect(md_sample, prints=verbose)
                
                
        print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
        print('\tThreshold: {:.2f}'.format(threshold), '\n')       
        
        
        for ind, md_sample in enumerate(x_md):
            detector.detect(md_sample, prints=True) 
            
        return m_d, detector, threshold, deg_start_indx
    
    
    def relative_to_start_period(self, x):
        return x/np.mean(x[:self.start_period])
