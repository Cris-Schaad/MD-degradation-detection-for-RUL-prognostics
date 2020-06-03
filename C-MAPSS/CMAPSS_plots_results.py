import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from CMAPSS_utils import close_all
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model = 'CNN'
load_model = 5

results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)
close_all()



def longest_sample_RUL_prediction():
    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        
        res_rmse = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'))
        model_name = 'model_10_results.npz'
        res_data = np.load(os.path.join(results_dir, sub_dataset, model_name),allow_pickle=True)
        
        y_true = res_data['y_true']
        y_pred = res_data['y_pred']

        test_lens = [len(x) for i, x in enumerate(y_true)]
        max_len_sample_ind = np.argwhere(test_lens == np.max(test_lens))[0][0]
        sample_true = y_true[max_len_sample_ind]
        sample_pred = y_pred[max_len_sample_ind]
        sample_rmse = rmse_eval(sample_true, sample_pred)
        
        plt.figure()
        plt.plot(np.arange(1, len(sample_true)+1,1), sample_true[::-1], label='True RUL')    
        plt.plot(np.arange(1, len(sample_true)+1,1), sample_pred[::-1], 'r', label='Predicted RUL')
        plt.xlabel('Operational cycles since degradation start detection')
        plt.ylabel('Remaining Useful Life')
        plt.legend()
        plt.text(0.65, 0.7, 'Sample RMSE '+'{:.2f}'.format(sample_rmse), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.savefig(os.path.join('plots','RUL_results',sub_dataset+'_longest_sample_RUL.png'),
                    bbox_inches='tight', pad_inches=0)
# longest_sample_RUL_prediction()        


def end_RUL_prediction():
    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        res_rmse = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'))
        best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_results.npz'
        res_data = np.load(os.path.join(results_dir, sub_dataset, best_model_name),allow_pickle=True)
        
        y_true = np.asarray([i[0] for i in res_data['y_true']]).flatten()
        y_pred = np.asarray([i[0] for i in res_data['y_pred']]).flatten()
        sample_rmse = rmse_eval(y_true, y_pred)
        
        sorted_idx = np.argsort(y_true)
        y_true = y_true[sorted_idx]
        y_pred = y_pred[sorted_idx]
        
        plt.figure()
        plt.plot(np.arange(1, len(y_true)+1,1), y_true[::-1], label='True RUL')    
        plt.plot(np.arange(1, len(y_true)+1,1), y_pred[::-1], 'r', label='Predicted RUL')
        plt.xlabel('Engine (sorted by RUL)')
        plt.ylabel('Remaining Useful Life')
        plt.legend()
        plt.text(0.75, 0.7, 'RMSE '+'{:.2f}'.format(sample_rmse), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.savefig(os.path.join('plots','RUL_results',sub_dataset+'_end_RUL.png'),
                    bbox_inches='tight', pad_inches=0)
# end_RUL_prediction()
           
            
def dataset_RUL_prediction():
    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        res_rmse = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'))
        best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_results.npz'
        res_data = np.load(os.path.join(results_dir, sub_dataset, best_model_name),allow_pickle=True)
        
        y_true = res_data['y_true']
        y_pred = res_data['y_pred']

        plt.figure()
        for i in range(len(y_true)):
            plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_true[i], 'C0')    
            plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_pred[i], 'r')
        plt.xlabel('Engine (sorted by RUL)')
        plt.ylabel('Remaining Useful Life')
        plt.legend()
        # plt.text(0.75, 0.7, 'RMSE '+'{:.2f}'.format(sample_rmse), 
        #          fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.savefig(os.path.join('plots','RUL_results',sub_dataset+'_all_RUL.png'),
                    bbox_inches='tight', pad_inches=0)
dataset_RUL_prediction()