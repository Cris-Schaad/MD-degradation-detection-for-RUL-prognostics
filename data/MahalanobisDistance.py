import numpy as np
from scipy.spatial.distance import mahalanobis


class MahalanobisDistance():
    
    def __init__(self, feature_axis=0, mode='inv'):
        self.mode = mode
        self.axis = feature_axis
        self.ms = False

    
    def fit_predict(self, x):
        """ Defines a Mahalanobis space"""
        
        md, u, s, cov = self.mahalanobis_distance(x)
        self.var_mean = u
        self.var_std = s
        self.cov = cov
        self.ms = True
        
        return md
        
    
    def fit(self, x):    
        """ Calculates distance to the defined Mahalanobis space"""
        
        if self.ms:
            md, _, _, _ = self.mahalanobis_distance(x, mean=self.var_mean, 
                                                    std=self.var_std, 
                                                    cov=self.cov)
            return md
        else:
            print('Mahalanobis space not constructed')
            
    
    def mahalanobis_distance(self, x, mean=None, std=None, cov=None):
        
        n_samples = x.shape[0]
        n_vars = x.shape[1]
        
        if mean is None:
            mean = np.mean(x, axis=self.axis)
        if std is None:
            std = np.std(x, axis=self.axis)
            
            
        if self.mode == 'covariance':
            if cov is None:    
                cov = self.covariance_matrix(x)  

            md = np.zeros(n_samples)    
            for i in range(len(x)):
                x_i = x[i,:] - mean

                MD = np.matmul(np.linalg.inv(cov), x_i)
                md[i] = np.sqrt(np.dot(np.transpose(x_i), MD)/(n_vars-1))
                
    
        if self.mode == 'correlation':
            z = np.ones_like(x)
            for i in range(n_vars):
                z[:,i] = (x[:,i]-mean[i])/std[i]
                
            if cov is None:    
                cov = self.correlation_matrix(z)       

            md = np.zeros(n_samples)    
            for i in range(len(z)):
                z_i = z[i,:]
                
                MD = np.matmul(np.linalg.inv(cov), z_i)
                md[i] = np.sqrt(np.dot(np.transpose(z_i), MD)/(n_vars-1))
        return md, mean, std, cov
            
                
    def correlation_matrix(self, x):
        n_samples = x.shape[0]
        n_vars = x.shape[1]
        
        cov = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                mean_i = np.mean(x[:,i])
                mean_j = np.mean(x[:,j])
                
                sup_sum = 0; i_sum = 0; j_sum = 0
                for k in range(n_samples):
                    sup_sum = sup_sum + (x[k,i]-mean_i)*(x[k,j]-mean_j)
                    i_sum = i_sum + (x[k,i]-mean_i)**2
                    j_sum = j_sum + (x[k,j]-mean_j)**2                    
                cov[i,j] = sup_sum/np.sqrt(i_sum*j_sum)
        return cov
    

    def covariance_matrix(self, x):
        n_vars = x.shape[1]
        cov = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                x_i = x[:,i] - np.mean(x[:,i])
                x_j = x[:,j] - np.mean(x[:,j])
                cov[i,j] = np.mean(x_i*x_j)
        return cov