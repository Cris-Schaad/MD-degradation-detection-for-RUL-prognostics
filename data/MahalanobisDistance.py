import numpy as np

class MahalanobisDistance():
    
    def __init__(self, feature_axis=0, mode='covariance'):
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
                x_i = (x[i,:] - mean)
                md[i] = np.sqrt(np.dot(np.transpose(x_i), np.matmul(np.linalg.inv(cov), x_i))/n_vars)
    
        if self.mode == 'correlation':
            z = np.ones_like(x)
            for i in range(n_vars):
                z[:,i] = (x[:,i]-mean[i])/std[i]
                
            if cov is None:    
                cov = self.correlation_matrix(x)       

            md = np.zeros(n_samples)    
            for i in range(len(z)):
                z_i = z[i,:]
                
                MD = np.matmul(np.linalg.inv(cov), z_i)
                md[i] = np.sqrt(np.dot(np.transpose(z_i), MD)/n_vars)
        return md, mean, std, cov
       
        
    def covariance(self, x, y):
        n = len(x)
        return np.dot(x-np.mean(x), y-np.mean(y))/(n-1)
    
    
    def covariance_matrix(self, x):
        n_vars = x.shape[1]
        cov = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                cov[i,j] = self.covariance(x[:,i], x[:,j])
        return cov
          
    
    def correlation(self, x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        x_sum = np.sum((x-mean_x)**2)
        y_sum = np.sum((y-mean_y)**2)
        return np.dot(x-mean_x, y-mean_y)/np.sqrt(x_sum*y_sum)
    
    
    def correlation_matrix(self, x):
        n_vars = x.shape[1]    
        corr = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):                 
                corr[i,j] = self.correlation(x[:,i], x[:,j])
        return corr