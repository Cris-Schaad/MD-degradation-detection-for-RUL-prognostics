import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf      
 

def program_start():
    plt.close('all')
    tf.reset_default_graph()

def plt_close():
    plt.close('all')
    
def save_folder(save_dir):
    if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    return None


def prediction_plots(y_true, y_pred, plot_name, save_dir=None, xlabel=None, ylabel=None):
    
    Y = np.column_stack((y_true, y_pred))
    Y = Y[Y[:,0].argsort()]
    err = abs(Y[:,0]-Y[:,1])
        
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    plt.figure(figsize=(6,6))
    
    plt.subplot(gs[0])
    plt.plot(range(1,len(Y)+1),Y[:,0], label='True RUL')
    plt.plot(range(1,len(Y)+1),Y[:,1], color='red', marker='.', linewidth=0.5, markersize=4, label='Predicted RUL')
    plt.xlim((0,len(Y)))
    plt.ylim((0, np.max(Y)))
    plt.ylabel(ylabel)
    plt.legend()
    
    plt.subplot(gs[1])
    plt.plot(range(1,len(Y)+1), err, label='Absolute error', color='black', linewidth=0.5)
    plt.xlabel(xlabel)
    plt.xlim((0,len(Y)))
    plt.ylabel('Absolute error')

    if save_dir:
        plt.savefig(os.path.join(save_dir,plot_name+'.svg'))
    return None


def loss_plot(training_loss, velidation_loss, save_dir=False, plot_name=None):
    plt.figure()
    plt.plot(training_loss, label='Entrenamiento')
    plt.plot(velidation_loss, label='Validación')
    plt.ylabel('Costo')
    plt.xlabel('Épocas')
    plt.legend(loc='upper right')
    if save_dir:
        plt.savefig(os.path.join(save_dir,plot_name+'.svg'))
    return None


def print_rmse(y_true, y_pred, test_name):
        test_loss = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))    
        print(test_name+' test RMSE: ', "{:.2f}".format(test_loss))
        return None

    
def save_results(y_true, y_pred, save_dir, time=None, name=''):
    np.savez(os.path.join(save_dir, name+'_results.npz'),
             y_true = y_true,
             y_pred = y_pred,
             training_time=time)
    return None


class MinMaxScaler():
    def __init__(self, feature_range=(0,1), feature_axis=1):
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.feature_axis = feature_axis
    
    def fit_transform(self, x_data):
        
        t_data = []
        x_data_min = []
        x_data_max = []

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