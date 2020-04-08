import numpy as np
from scipy.spatial.distance import mahalanobis


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
            
        if self.mode == 'scipy':
            if cov is None:    
                cov = np.cov(x, rowvar=False)         

            md = np.zeros(n_samples)    
            for i in range(len(z)):
                x_i = x[i,:]                
                md[i] = mahalanobis(x_i, mean, np.linalg.inv(cov))/n_vars
    
        if self.mode == 'inv':
            if cov is None:    
                cov = self.correlation_matrix(z)           

            md = np.zeros(n_samples)    
            for i in range(len(z)):
                z_i = z[i,:]
                
                MD = np.matmul(np.linalg.inv(cov), z_i)
                md[i] = np.dot(np.transpose(z_i), MD)/n_vars
    
        if self.mode == 'GS':
            U = np.zeros_like(z)
            for i in range(n_vars):
                if i == 0:
                    U[:,i] = z[:,i]
                else:
                    proyections = [U[:,k]*np.dot(z[:,i],U[:,k])/np.dot(U[:,k], U[:,k]) for k in range(i)]
                    U[:,i] = z[:,i] - np.sum(proyections, axis=0)
           
            s = np.std(U, axis=axis)
            md = np.zeros(n_samples)    
            for i in range(len(z)):
                md[i] = np.mean([np.power(U[i,k]/s[k], 2) for k in range(n_vars)])
        
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
    
    
    def mahalanobis_space(self, md):
        return None