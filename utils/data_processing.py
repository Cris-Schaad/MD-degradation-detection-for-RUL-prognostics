import os
import numpy as np  
from scipy.stats import skew, kurtosis



class MinMaxScaler():
    def __init__(self, feature_range=(0,1), feature_axis=1):
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.feature_axis = feature_axis
    
    def fit_transform(self, x_data):
        
        t_data = []
        x_data_min = []
        x_data_max = []

        print('Number of independent variables to scale: ', x_data.shape[self.feature_axis])
        for i in range(x_data.shape[self.feature_axis]):
            x = x_data.take(i, axis=self.feature_axis)
            x_min = np.min(x)
            x_max = np.max(x)
            x_data_min.append(x_min)
            x_data_max.append(x_max)
            
            if x_min == x_max:
                t_data.append(np.expand_dims(np.ones(x.shape), axis=self.feature_axis))
            else:
                data_scaled = (x - x_min)*((self.max - self.min)/(x_max - x_min)) + self.min
                t_data.append(np.expand_dims(data_scaled, axis=self.feature_axis))  
                
        self.x_data_min = np.asarray(x_data_min)
        self.x_data_max = np.asarray(x_data_max)
        return  np.concatenate(t_data, axis=self.feature_axis)

    def transform(self, x_data):
        
        t_data = []
        for i in range(x_data.shape[self.feature_axis]):
            x = x_data.take(i, axis=self.feature_axis)            
            
            if self.x_data_min[i] == self.x_data_max[i]:
                data_scaled = x/self.x_data_max[i]
                t_data.append(np.expand_dims(np.ones(x.shape), axis=self.feature_axis))
            else:
                data_scaled = (x - self.x_data_min[i])*((self.max - self.min)/(self.x_data_max[i] - self.x_data_min[i])) + self.min
                t_data.append(np.expand_dims(data_scaled, axis=self.feature_axis))  
                
        return np.concatenate(t_data, axis=self.feature_axis)
        
    def inverse_transform(self, t_data):

        x_data = []
        for i in range(t_data.shape[self.feature_axis]):
            t = t_data.take(i, axis=self.feature_axis)
            data_scaled = (t - self.min)*((self.x_data_max[i] - self.x_data_min[i])/(self.max - self.min)) + self.x_data_min[i]
            x_data.append(np.expand_dims(data_scaled, axis=self.feature_axis))        
        
        return np.concatenate(x_data, axis=self.feature_axis) 



def time_window_sampling(x, y, time_window, 
                         temporal_axis=0, 
                         time_step = 1, 
                         y_time_lag = 0, 
                         add_last_dim = False, 
                         time_step_flatten = False):
    framed_x = []
    framed_y = []
    for x_sample, y_sample in zip(x, y):
        
        indx = x_sample.shape[temporal_axis] - y_time_lag
        indx_left = True if indx - time_window >= 0 else False
        
        if indx_left == True:
            framed_x_sample = []
            framed_y_sample = []
            
            x_sample = np.swapaxes(x_sample, 0, temporal_axis)
            while indx_left:
                
                time_step_data = x_sample[indx-time_window:indx] 
                time_step_data = np.swapaxes(time_step_data, temporal_axis, 0)
                if time_step_flatten:
                    time_step_data = time_step_data.flatten()
                    
                framed_x_sample.append(time_step_data)
                framed_y_sample.append(y_sample[indx+y_time_lag-1])
                indx = indx-time_step

                if indx-time_window <= 0:
                    indx_left=False
            
            if add_last_dim: 
                framed_x_sample = np.expand_dims(np.asarray(framed_x_sample), axis=-1)
            else:
                framed_x_sample = np.asarray(framed_x_sample)
            
            if time_window ==1:
                framed_x_sample = np.squeeze(framed_x_sample, axis=1)
            
            framed_x.append(framed_x_sample)
            framed_y.append(np.asarray(framed_y_sample))
    if len(framed_x) < len(x):
        print('Samples ignored:', len(x)-len(framed_x))        
    
    return np.asarray(framed_x), np.asarray(framed_y)



def windows(x, time_window, time_step):
    
    wins = []
    indx = len(x)
    indx_left = True
    while indx_left:
        sample = x[indx-time_window:indx]
        wins.append(sample[::-1])
        indx = indx-time_step
        if indx-time_window < 0:
            indx_left=False
    return np.asarray(wins[::-1])



def samples_under_deg(x, ind):
    x_not_under_deg = []
    x_under_deg = []
    for i in range(len(x)):
        if ind[i] == len(x[i])-1:
            x_not_under_deg.append(x[i])
        else:
            x_under_deg.append(x[i])
    return np.asarray(x_not_under_deg), np.asarray(x_under_deg)



def samples_reverse(x):
    for i in range(len(x)):
        x[i] = x[i][::-1]
    return x



def features(x):
    rms = np.sqrt(np.mean(x**2, axis=0))
    ptp = np.max(x) - np.min(x)
    cf = abs(ptp)/rms
    mean = np.mean(x, axis=0)
    maxi = np.max(abs(x), axis=0)
    var = np.var(x, axis=0)
    skw = skew(x, axis=0)
    kurt = kurtosis(x, axis=0)

    output_features = np.column_stack((rms,ptp,cf,mean,maxi,var,skw,kurt))
    return output_features.flatten()