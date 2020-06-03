import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
 

class ResultsSaver():
    def __init__(self, parent_dir, folder_name, file_name, file_type='.csv'):
        
        self.save_folder_dir = os.path.join(parent_dir, folder_name)
        if os.path.exists(self.save_folder_dir):
                shutil.rmtree(self.save_folder_dir)
        if not os.path.exists(self.save_folder_dir):
            os.makedirs(self.save_folder_dir)
            
        self.save_file = os.path.join(self.save_folder_dir, file_name+file_type)
        with open(self.save_file, 'w') as f:
            f.write('Iter, RMSE\n')
            f.close()
            
    def save_iter(self, iter_, rmse):
        with open(self.save_file, 'a') as f:
            f.write(str(iter_)+','+str(rmse)+'\n')
            f.close()
        

def prediction_plots(y_true, y_pred, plot_name="", save_dir=None, xlabel=None, ylabel=None):
    
    Y = np.column_stack((y_true, y_pred))
    Y = Y[Y[:,0].argsort()[::-1]]
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


def rmse_eval(y_true, y_pred, test_name=None):
        rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))    
        if test_name is not None:
            print(test_name+' RMSE: {:.2f}'.format(rmse))
        return rmse