import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from CMAPSS_utils import close_all
from utils.training_functions import prediction_plots
from utils.training_functions import rmse_eval


dataset_name = 'CMAPSS'
model = 'CNN'

results_dir = os.path.join(os.getcwd(), 'training_results', dataset_name, model)
close_all()


for sub_dataset in ['FD001','FD002','FD003','FD004']:
    saved_results_list = [i for i in os.listdir(os.path.join(results_dir, sub_dataset)) if '.npz' in i]
    res_data = np.load(os.path.join(results_dir, sub_dataset, saved_results_list[0]), allow_pickle=True)
    
    y_true = res_data['y_true']
    y_pred = res_data['y_pred']

    # prediction_plots(y_test, y_pred, plot_name=sub_dataset)    
    # test_loss = rmse_eval(y_test, y_pred, sub_dataset)
    
    
    test_lens = [len(x) for i, x in enumerate(y_true)]
    min_len_sample_ind = np.argwhere(test_lens == np.min(test_lens))[0][0]
    max_len_sample_ind = np.argwhere(test_lens == np.max(test_lens))[0][0]
    
    
    plt.figure()
    for i in range(len(y_true)):
        plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_true[i])    
        plt.plot(np.arange(np.min(y_true[i]),np.max(y_true[i])+1,1), y_pred[i])