import numpy as np


class MahalanobisDistance():
    
    def __init__(self, feature_axis=0, mode='inv'):
        self.mode = mode
        self.axis = feature_axis


    def fit_predict(self, x):
        md, u, s, cov = self.mahalanobis_distance(x, axis=self.axis)
        self.ms = md
        self.var_mean = u
        self.var_std = s
        self.cov = cov
        return md
        
    
    def fit(self, x):    
        md, _, _, _ = self.mahalanobis_distance(x, mean=self.var_mean, std=self.var_std, cov=self.cov)
        return md
    
    
    def mahalanobis_distance(self, x, mean=None, std=None, cov=None, axis=0):
        
        n_samples = x.shape[0]
        n_vars = x.shape[1]
        
        if mean is None:
            mean = np.mean(x, axis=axis)
        if std is None:
            std = np.std(x, axis=axis)
        
        z = np.ones_like(x)
        for i in range(n_vars):
            z[:,i] = (x[:,i]-mean[i])/std[i]
            
        if self.mode == 'inv':
            if cov is None:    
                cov = np.matmul(np.transpose(z), z)/(n_vars)
            
            md = np.zeros(n_samples)    
            for i in range(len(z)):
                z_i = z[i,:]
                
                MD = np.matmul(np.linalg.inv(cov), z_i)/n_vars
                md[i] = np.dot(np.transpose(z_i), MD)
    
        if self.mode == 'GS':
            U = np.zeros_like(z)
            for i in range(n_vars):
                if i == 0:
                    U[:,i] = z[:,i]
                else:
                    a = 0
                    for k in range(1,i):
                        a = a + U[:,k-1]*np.dot(np.transpose(z[:,k]), U[:,k-1])/np.dot(np.transpose(U[:,k-1]), U[:,k-1])
                    U[:,i] = z[:,i] - a
           
            s = np.std(U, axis=axis)
            md = np.zeros(n_samples)    
            for i in range(len(z)):
                a = 0
                for k in range(n_vars):
                    a = a + (U[i,k]**2)/(s[k]**2)
                md[i] = a/n_vars
        
        return md, mean, std, cov
            
                
    
    def mahalanobis_space(self, md):
        return None