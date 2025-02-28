import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from CMAPSS_utils import close_all
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model = 'CNN'

results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)
close_all()


def RUL_RMSE_stats_non_degraded_samples():
    dataset_name = 'CMAPSS_unfiltered'
    results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)

    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        
        rmse = []
        runs = [i for i in os.listdir(os.path.join(results_dir, sub_dataset)) if 'ignored' in i]
        for run in runs:
            data = np.load(os.path.join(results_dir, sub_dataset, run), allow_pickle=True)
            y_true = np.asarray([int(i[0]) for i in data['y_true']])
            y_pred = np.asarray([int(i[0]) for i in data['y_pred']])
            rmse.append(rmse_eval(y_true, y_pred))
        print(sub_dataset+' mean RMSE: {:.2f}'.format(np.mean(rmse)))
        print(sub_dataset+' std. RMSE: {:.2f}'.format(np.std(rmse)))        
# RUL_RMSE_stats_non_degraded_samples()



def RUL_RMSE_stats():
    dataset_name = 'CMAPSS'
    results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)

    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        
        data = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'), sep=',')
        print(sub_dataset+' mean RMSE: {:.2f}'.format(np.mean(data[' RMSE'])))
        print(sub_dataset+' std. RMSE: {:.2f}'.format(np.std(data[' RMSE'])))        
# RUL_RMSE_stats()  
    
    
    

def longest_sample_RUL_prediction():
    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        
        res_rmse = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'))
        best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_results.npz'
        res_data = np.load(os.path.join(results_dir, sub_dataset, best_model_name),allow_pickle=True)
        
        y_true = res_data['y_true']
        y_pred = res_data['y_pred']
                
        md_data = np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True)[sub_dataset][()]
        x_md = md_data['x_test_md']
        x_ds = md_data['test_deg']
        test_lens = [len(x) for i, x in enumerate(x_md) if x_ds[i] < len(x) -1]

        
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


def FD004_cases_RUL():
        
    res_rmse = pd.read_csv(os.path.join(results_dir, 'FD004', 'FD004.csv'))
    best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_results.npz'
    res_data = np.load(os.path.join(results_dir, 'FD004', best_model_name),allow_pickle=True)
    
    y_true = res_data['y_true']
    y_pred = res_data['y_pred']
            
    md_data = np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True)['FD004'][()]
    x_md = md_data['x_test_md']
    x_ds = md_data['test_deg']
    threshold = md_data['threshold']
    md_ind = [i for i, x in enumerate(x_md) if x_ds[i] < len(x) -1]  


    indexes = []
    # Longest RUL sample
    test_lens =  [len(y) for y in y_pred]
    indexes.append(np.argwhere(test_lens == np.max(test_lens))[0][0])
    
    # Random regeneration
    pred_ruls = [np.min(y[150:]) if len(y)>150 else np.inf for y in y_pred]
    indexes.append(np.argwhere(pred_ruls == np.min(pred_ruls))[0][0])
    
    for ind in indexes:
        sample_true = y_true[ind]
        sample_pred = y_pred[ind]
        sample_rmse = rmse_eval(sample_true, sample_pred)

        md_sample = x_md[md_ind[ind]]
        no_deg_sample = md_sample[:x_ds[md_ind[ind]]]
        deg_sample = md_sample[x_ds[md_ind[ind]]:]
        
        fig, ax = plt.subplots(2,1, figsize=(7,7))
        ax.flat
                
        ax[0].plot(x_ds[md_ind[ind]]+np.arange(1, len(sample_true)+1,1), sample_true[::-1], label='True RUL')    
        ax[0].plot(x_ds[md_ind[ind]]+np.arange(1, len(sample_true)+1,1), sample_pred[::-1], 'r', label='Predicted RUL')
        ax[0].set_xlim(xmin=-5)
        ax[0].set_ylabel('RUL')       
        ax[0].legend()
        
        ax[1].plot(range(len(no_deg_sample)), no_deg_sample, color='g')
        ax[1].plot(range(len(no_deg_sample), len(no_deg_sample)+len(deg_sample)), deg_sample, color='r')  
        ax[1].axhline(threshold, c='k', label='MD threshold')
        ax[1].set_xlim(xmin=-5)
        ax[1].set_xlabel('Operational cycles')
        ax[1].set_ylabel('MD')
        ax[1].legend()

        fig.savefig(os.path.join('plots','FD004_samples_{}.png'.format(md_ind[ind]+1)),
                    bbox_inches='tight', pad_inches=0)
FD004_cases_RUL()      


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
        plt.xlabel('Remaining Useful Life')
        plt.ylabel('Remaining Useful Life')
        
        if sub_dataset=='FD004':
            plt.xlim(0,300)
            plt.ylim(0,300)
        
        plt.legend(('True RUL', 'Predicted RUL'), loc='upper right', framealpha=1)
        ax = plt.gca()
        ax.invert_xaxis()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('C0')
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[0].set_linewidth(2)
        leg.legendHandles[1].set_color('r')
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[1].set_linewidth(2)

        plt.savefig(os.path.join('plots','RUL_results',sub_dataset+'_all_RUL.png'),
                    bbox_inches='tight', pad_inches=0)
# dataset_RUL_prediction()


def dataset_unfiltered_RUL_prediction():
    dataset_name = 'CMAPSS_unfiltered'
    results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)
    
    for sub_dataset in ['FD001','FD002','FD003','FD004']:
        res_rmse = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'))
        best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_ignored_samples_results.npz'
        res_data = np.load(os.path.join(results_dir, sub_dataset, best_model_name),allow_pickle=True)
        
        y_true = res_data['y_true']
        y_pred = res_data['y_pred']

        plt.figure()
        for i in range(len(y_true)):
            plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_true[i], 'C0')    
            plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_pred[i], 'r')
        plt.xlabel('Remaining Useful Life')
        plt.ylabel('Remaining Useful Life')
        
        plt.legend(('True RUL', 'Predicted RUL'), loc='upper right', framealpha=1)
        ax = plt.gca()
        ax.invert_xaxis()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('C0')
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[0].set_linewidth(2)
        leg.legendHandles[1].set_color('r')
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[1].set_linewidth(2)

        plt.savefig(os.path.join('plots','RUL_results',sub_dataset+'_undegraded_samples_RUL.png'),
                    bbox_inches='tight', pad_inches=0)
# dataset_unfiltered_RUL_prediction()