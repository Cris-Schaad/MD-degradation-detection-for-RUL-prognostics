import numpy as np
import matplotlib.pyplot as plt

from MahalanobisDistance import MahalanobisDistance


class Detector():

    def __init__(self, k, n):
        self.k = k
        self.n = n

    def detect(self, x, threshold, verbose=False):
        for i in range(self.n-1, len(x)):
            sample_k = 0
            for m in range(self.n): 
                if x[i - (self.n-1) + m] >= threshold:
                    sample_k = sample_k +1
                
            if sample_k >= self.k:
                return i
            
            if i == len(x)-1:
                if verbose:
                    print('No tresspassing detected')
                return i
    
    def sampling_from_index(self, x, y, ind, time_window):
        x_sampled = []
        y_sampled = []
        for x_sample, y_sample, deg_start_index in zip(x, y, ind):
            if deg_start_index < len(x_sample)-1:
                ind_start = deg_start_index-time_window if deg_start_index-time_window > 0 else 0
                x_sampled.append(x_sample[ind_start:])
                y_sampled.append(y_sample[ind_start:]) 
        return np.asarray(x_sampled), np.asarray(y_sampled)



class iterator():
    
    def __init__(self, k, n, sigma,
                 tolerance=0.00001,
                 max_iter=100):
        """
        k: number of conditions out of "n" consecutive points that are out of Mahalanobis Space
        n: number of consecutive conditions
        sigma: standard deviation that defines the Mahalanobis Space
        """
        
        self.m_d = MahalanobisDistance(mode='covariance')
        self.detector = Detector(k, n)
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
                    self.threshold = threshold
                    break
            
            if i == self.max_iter-1:
                print('Max iterations ({:}) reached'.format(self.max_iter))
         
        print('\tCovergence at iter: {}'.format(len(self.iter_ms_dim)))                     
        print('\tMean MD in MS: {:.2f}'.format(np.mean(ms)))
        print('\tThreshold: {:.2f}\n'.format(threshold))       
        
        for ind, md_sample in enumerate(x_md):
            self.detector.detect(md_sample, threshold, verbose=True)    
        return deg_start_indx, threshold
    
    
    def md_calculation_op(self, x): 
        return np.asarray([self.m_d.fit(sample) for sample in x])
    
    
    def degradation_start_index_from_MD(self, x, verbose):
        return [self.detector.detect(x_sample, self.threshold, verbose=verbose) for x_sample in x]
    
    
    def sampling_from_MD_deg_start(self, x, y, extra_indexes=0, verbose=False):
        x_md = self.md_calculation_op(x)
        deg_start_indx = self.degradation_start_index_from_MD(x_md, verbose)
        return self.detector.sampling_from_index(x, y, deg_start_indx, extra_indexes)
    
    
    def plot_RUL_after_deg(self, x, y, plot=True):
        
        x_md = self.md_calculation_op(x)
        deg_start_indx = self.degradation_start_index_from_MD(x_md, verbose=False)
                
        samples_rul = []
        for i in range(len(y)):
            if deg_start_indx[i] < len(y[i]) -1:
                samples_rul.append(y[i][deg_start_indx[i]][0])
                
        print('Average RUL: {:.2f}'.format(np.mean(samples_rul)))
        print('Min RUL: {:.2f}'.format(np.min(samples_rul)))
        print('Max RUL: {:.2f}\n'.format(np.max(samples_rul))) 
        
        if plot:
            plt.figure()
            plt.hist(samples_rul, 20, range=(0,300), alpha=0.5)
        
        
    def plot_lifetime_dist(self, x, y):

        x_md = self.md_calculation_op(x)
        deg_start_indx = self.degradation_start_index_from_MD(x_md, verbose=False)
                        
        samples_with_degradation = []
        samples_without_degradation = []
        for i in range(len(y)):
            if deg_start_indx[i] < len(y[i]) -1:
                samples_with_degradation.append(y[i])
            else:
                samples_without_degradation.append(y[i])
        
        print('Total samples left in dataset: {}'.format(len(samples_with_degradation)))                
        print('Minimum sample length left in dataset: {}\n'.format(np.min([len(i) for i in samples_with_degradation])))
       
        plt.figure()
        plt.hist([np.min(i) for i in samples_with_degradation], 20, range=(0,200), color='r', alpha=0.5)
        plt.hist([np.min(i) for i in samples_without_degradation], 20, range=(0,200), color='g', alpha=0.5)

        
        
        
        
        
        
