import numpy as np

import MahalanobisDistance as MD
import degradation_detection as dd


class iterator():
    
    def __init__(self, k, n, sigma,
                 moving_window=1, 
                 tolerance=0.00001,
                 max_iter=100):
        """
        k: number of conditions out of "n" consecutive points that are out of Mahalanobis Space
        n: number of consecutive conditions
        sigma: standard deviation that defines the Mahalanobis Space
        """
        
        self.m_d = MD.MahalanobisDistance(mode='covariance')
        self.detector = dd.Detector(k, n, moving_window)
        self.s = sigma
        
        self.iter_ms_dim = []
        self.tol = tolerance
        self.max_iter = max_iter        
        
        
    def iterative_calculation(self, x, n_iterations=10, verbose=True):

        deg_start_indx = [i.shape[0]-1 for i in x]
        for i in range(self.max_iter):
            if verbose:
                print('\tIteration: {:}'.format(int(i+1)))
                            
            # MD calculation
            x_healthy = np.asarray([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x)])    
            self.m_d.fit_predict(np.concatenate(x_healthy))
            x_md = self.md_calculation_op(x)    
            
            # MS and threshold
            ms = np.concatenate([sample[:deg_start_indx[ind]] for ind, sample in enumerate(x_md)])
            threshold = np.mean(ms) + self.s*np.std(ms)
            
            # Degradation start detection
            for ind, md_sample in enumerate(x_md):
                deg_start_indx[ind] = self.detector.detect(md_sample, threshold)
            
            # Algorithm convergence check
            self.iter_ms_dim.append(len(ms))            
            if i > 0: 
                if (self.iter_ms_dim[i-1] - self.iter_ms_dim[i])/self.iter_ms_dim[i-1] <= self.tol:
                    break
            
            if i == self.max_iter-1:
                print('Max iterations ({:}) reached'.format(self.max_iter))
         
        print('\tCovergence at iter: {}'.format(len(self.iter_ms_dim)))                     
        print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
        print('\tThreshold: {:.2f}'.format(threshold))       
        print('\tCov determinant {:.6f}'.format(np.linalg.det(self.m_d.cov)), '\n')
        
        for ind, md_sample in enumerate(x_md):
            self.detector.detect(md_sample, threshold, verbose=True) 
            
        return deg_start_indx, threshold
    
    
    def md_calculation_op(self, x): 
        return np.asarray([self.m_d.fit(sample) for sample in x])
   
    
    def get_detector(self):
        return self.detector
