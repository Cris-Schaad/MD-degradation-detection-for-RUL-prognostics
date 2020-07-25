import os
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
    
    def dataset_detect(self, x, threshold):
        indx = []
        for sample in x:
            indx.append(self.detect(sample, threshold))
        return np.asarray(indx)
        


class MSIterativeAlgorithm():
    
    def __init__(self, k, n, sigma,
                 tolerance=0,
                 max_iter=500):
        """
        k: number of conditions out of "n" consecutive points that are out of Mahalanobis Space
        n: number of consecutive conditions
        sigma: standard deviation that defines the Mahalanobis Space
        """
        
        self.m_d = MahalanobisDistance(mode='covariance')
        self.detector = Detector(k, n)
        self.sigma = sigma
        
        self.iter_ms_dim = []
        self.tol = tolerance
        self.max_iter = max_iter        
        
        
    def iterative_calculation(self, x, n_iterations=10, verbose=True, plot_dist_change=True):

        # Initialy all instances are in the MS
        ms_index = [np.arange(0, len(sample), 1) for sample in x]        

        # Iterativa algorithm
        for i in range(self.max_iter):
            if verbose:
                print('\tIteration: {:}'.format(int(i+1)))
                            
            # MD calculation
            x_healthy = np.asarray([sample[ms_index[ind]] for ind, sample in enumerate(x)])                
            self.m_d.fit_predict(np.concatenate(x_healthy))
            x_md = self.md_calculation_op(x)    
            ms = np.concatenate([sample[ms_index[ind]] for ind, sample in enumerate(x_md)])
            
            # MS to normal distribution
            self.ms_log_mean = np.mean(np.log(ms))
            self.ms_log_std = np.std(np.log(ms))
            
            # Statistical threshold 
            self.log_threshold = self.sigma*self.ms_log_std
            self.threshold = np.exp(self.log_threshold*self.ms_log_std + self.ms_log_mean)
            
            # Degradation start detection
            for ind, md_sample in enumerate(x_md):
                deg_start = self.detector.detect(self.log_normalize(md_sample), self.log_threshold)
                ms_index[ind] = np.arange(0, deg_start, 1)              
                
            # Algorithm convergence check
            self.iter_ms_dim.append(len(ms))     
            if i == 0:
                ms_original = ms
            if i > 0: 
                if np.abs((self.iter_ms_dim[i-1] - self.iter_ms_dim[i]))/self.iter_ms_dim[i-1] <= self.tol:
                    break
            if i == self.max_iter-1:
                print('Max iterations ({:}) reached'.format(self.max_iter))
        
        # Distribution plots
        if plot_dist_change:
            plt.figure()    
            lim = 1.1*np.max([np.max(np.abs(self.log_normalize(ms))), 
                              np.max(np.abs(self.log_normalize(ms_original)))])
            plt.hist(self.log_normalize(ms_original),
                      bins=np.linspace(-lim,lim,500), color='C0', alpha=0.3, density=True, label='Initial distribution')
            plt.hist(self.log_normalize(np.concatenate([sample[ms_index[ind]] for ind, sample in enumerate(x_md)])),
                      bins=np.linspace(-lim,lim,500), color='C2', alpha=0.3, density=True, label='MS')
            plt.hist(self.log_normalize(np.concatenate([sample[ms_index[ind][-1]:] for ind, sample in enumerate(x_md)])), 
                      bins=np.linspace(-lim,lim,500), color='r', alpha=0.3, density=True, label='MS outsiders')
            plt.legend()
            plt.savefig(os.path.join('plots', 'MS_dist_change'), bbox_inches='tight', pad_inches=0)
        
        # Output prints
        print('\tCovergence at iter: {}'.format(len(self.iter_ms_dim)))                     
        print('\tMS proportion: {:.2f}%'.format(100*self.iter_ms_dim[-1]/len(np.concatenate(x))))       
        print('\tMD threshold: {:.2f}\n'.format(self.threshold))       
        
        # All samples cross threshold check
        for ind, md_sample in enumerate(x_md):
            self.detector.detect(self.log_normalize(md_sample), self.log_threshold, verbose=True)    
        return None
    
    
    def log_normalize(self, x):
        return (np.log(x)-self.ms_log_mean)/self.ms_log_std
    
    
    def md_calculation_op(self, x): 
        return np.asarray([self.m_d.fit(sample) for sample in x])
    
    
    def detect_degradation_start_index_from_MD(self, x, verbose):
        return [self.detector.detect(self.log_normalize(x_sample), self.log_threshold, verbose=verbose) 
                for x_sample in x]
    
    
    def detect_degradation_start(self, x, verbose=False):
        x_md = self.md_calculation_op(x)
        return self.detect_degradation_start_index_from_MD(x_md, verbose=verbose)
    
    
    def RUL_info(self, x, y, plot=True):
        deg_start_indx = self.detect_degradation_start(x, verbose=False)
                
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
        
        
    def plot_lifetime_dist(self, x, y, savepath='', figname=''):
        deg_start_indx = self.detect_degradation_start(x, verbose=False)
              
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
        plt.hist([np.min(i) for i in samples_with_degradation], 20, 
                 range=(0,200), color='r', alpha=0.5, label='Samples reaching degradation')
        plt.hist([np.min(i) for i in samples_without_degradation], 
                 20, range=(0,200), color='g', alpha=0.5, label='Samples not reaching degradation')
        plt.xlabel('Engine total lifetime')
        plt.ylabel('Number of engines')
        plt.legend()
        plt.savefig(os.path.join(savepath, figname+'_testset_RUL_dist'), bbox_inches='tight', pad_inches=0)


        
        
        
        
        
        
