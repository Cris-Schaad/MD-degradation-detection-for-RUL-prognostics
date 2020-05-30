import os
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


def res_to_npz():
    res_folders = [i for i in os.listdir(os.getcwd()) if 'LSTM' in i and '.npz' not in i]
    for folder in res_folders:
        
        print(folder)
        res_dict = dict()
        for subdataset_folder in ['FD001', 'FD002', 'FD003', 'FD004']:
            folder_path = os.path.join(folder, subdataset_folder)
            npz_files = np.sort(np.asarray([i for i in os.listdir(os.path.join(folder, subdataset_folder)) if '.npz' in i]))
    
            all_iters_dict = dict()
            rmse = []
            for i, npz_name in enumerate(npz_files):
                subdataset_iter = np.load(os.path.join(folder_path, npz_name), allow_pickle=True)
                
                err = []
                for j in range(len(subdataset_iter['y_true'])):
                    err.append(np.power(subdataset_iter['y_true'][j][0] - subdataset_iter['y_pred'][j][0], 2))
                rmse.append(np.sqrt(np.mean(err)))

                if i == 0:
                    all_iters_dict['y_pred'] = subdataset_iter['y_pred']
                else:
                    stacked_data = []
                    for k in range(subdataset_iter['y_pred'].shape[0]):
                        stacked_data.append(np.column_stack((all_iters_dict['y_pred'][k], subdataset_iter['y_pred'][k])))
                    all_iters_dict['y_pred'] = np.asarray(stacked_data)
               
            if i > 0:
                for k in range(all_iters_dict['y_pred'].shape[0]):
                    all_iters_dict['y_pred'][k] = np.mean(all_iters_dict['y_pred'][k], axis=1)
            
            print(subdataset_folder)
            print('Mean RMSE :', "{:.2f}".format(np.mean(rmse)))
            print('Std. RMSE :', "{:.2f}".format(np.std(rmse)))
            
            subdataset_dict = {'y_true': subdataset_iter['y_true'],
                               'y_pred': all_iters_dict['y_pred']}
            res_dict[subdataset_folder] = subdataset_dict
            
        np.savez(folder,
                 **res_dict)
#res_to_npz()


def rul_ann_testset_comparisson():
    npz_files = ['LSTM_RUL_from_health_index_no_op_settings.npz']
    
    main_npz = np.load('LSTM_RUL_from_health_index.npz', allow_pickle=True)
    for sub_dataset in main_npz.keys():
        sub_dataset_dict = main_npz[sub_dataset][()]
        y_pred = sub_dataset_dict['y_pred']
        y_true = sub_dataset_dict['y_true']
        
        pred_y_samples = []
        true_y_samples = []        
        for i, sample in enumerate(y_pred):
            pred_y_samples.append(y_pred[i][0])
            true_y_samples.append(y_true[i][0][0])
        true_y_samples = np.asarray(true_y_samples)
           
        pred_y_comparisson = []
        for npz in npz_files:
            res_npz = np.load(npz, allow_pickle=True)[sub_dataset][()]
            model_to_compare = []
            for i in range(len(res_npz['y_pred'])):
                model_to_compare.append(res_npz['y_pred'][i][0])
            pred_y_comparisson.append(np.asarray(model_to_compare))
        pred_y_comparisson = np.asarray(pred_y_comparisson)
        pred_y_samples = np.asarray([np.asarray(pred_y_samples), np.asarray(pred_y_comparisson)])

        sorted_index = np.argsort(true_y_samples)[::-1]      
        plt.figure()
        plt.plot(true_y_samples[sorted_index], label='True label')
        
        colors = ['r', 'g']
        labels = ['RUL-ANN', 'RUL-ANN without op. settings']

        for i in range(len(pred_y_samples)):
            print('RMSE', labels[i], "{:.2f}".format(np.sqrt(np.mean(np.power(true_y_samples[sorted_index] - pred_y_samples[i].flatten()[sorted_index], 2)))))
            
        for i in [1,0]:
            plt.plot(pred_y_samples[i].flatten()[sorted_index], c=colors[i], label=labels[i], marker='.', linewidth=0.5, markersize=4)

        plt.xlabel('Test set sample (sorted from max. to min. RUL)')
        plt.ylabel('RUL')
        plt.legend(loc = 'upper right')
        plt.savefig(os.path.join('images','RUL_ANN_testset_'+sub_dataset), bbox_inches='tight', pad_inches=0)       
#rul_ann_testset_comparisson()



def rul_ann_sample_comparisson():
    npz_files = ['LSTM_RUL_from_health_index_no_op_settings.npz']
    
    main_npz = np.load('LSTM_RUL_from_health_index.npz', allow_pickle=True)
    for sub_dataset in main_npz.keys():
        sub_dataset_dict = main_npz[sub_dataset][()]
        y_pred = sub_dataset_dict['y_pred']
        y_true = sub_dataset_dict['y_true']
        
        #Max length         
        j = 0
        for i, sample in enumerate(y_pred):
            if len(sample) > j:
                j=len(sample)
                i_max = i
                
        pred_y_samples = []
        true_y_samples = []
        pred_y_samples.append(y_pred[i_max])
        true_y_samples.append(y_true[i_max])
        
        for npz in npz_files:
            res_npz = np.load(npz, allow_pickle=True)[sub_dataset][()]
            pred_y_samples.append(res_npz['y_pred'][i_max])
            true_y_samples.append(res_npz['y_true'][i_max])
            
        max_sample_len = np.max([len(i) for i in true_y_samples])
        y_true = true_y_samples[np.argwhere(max_sample_len == [len(i) for i in true_y_samples])[0][0]][::-1]
        x_true = np.linspace(0, max_sample_len, max_sample_len)
        
        plt.figure(figsize=(12,4))
        plt.plot(x_true, y_true, label='True label')
        colors = ['r','g']
        labels = ['RUL-ANN', 'RUL-ANN without op. settings']
        
        x_samples = []
        for i in true_y_samples:
            x_samples.append(np.linspace(max_sample_len, max_sample_len-len(i), len(i)))
            
        rmse = []
        last_error = []
        for i in range(len(true_y_samples)):
            plt.plot(x_samples[i], pred_y_samples[i], colors[i], label= labels[i])
            rmse.append("{:.2f}".format(np.sqrt(np.mean(np.power(y_true - pred_y_samples[i][::-1], 2)))))
            last_error.append("{:.2f}".format(pred_y_samples[i][0] - y_true[-1][0]))
        plt.xlabel('Flight cycles')
        plt.ylabel('RUL')
        
        plt.text(0.7, 0.65, 'RMSE: '+rmse[0], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        plt.text(0.7, 0.57, 'RMSE without op. settings: '+rmse[1], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        plt.text(0.7, 0.49, 'Last instance error: '+last_error[0], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        plt.text(0.7, 0.41, 'Last instance error without op. settings: '+last_error[1], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        
        plt.legend()
        plt.savefig(os.path.join('images','RUL_ANN_sample_'+sub_dataset), bbox_inches='tight', pad_inches=0)       
#rul_ann_sample_comparisson()


def model_sample_comparisson():
    npz_files = ['LSTM_RUL_from_degradation_start.npz', 'LSTM_RUL_complete.npz']
    
    main_npz = np.load('LSTM_RUL_from_health_index.npz', allow_pickle=True)
    for sub_dataset in main_npz.keys():
        sub_dataset_dict = main_npz[sub_dataset][()]
        y_pred = sub_dataset_dict['y_pred']
        y_true = sub_dataset_dict['y_true']
        
        #Max length         
        j = 0
        for i, sample in enumerate(y_pred):
            if len(sample) > j:
                j=len(sample)
                i_max = i
                
        pred_y_samples = []
        true_y_samples = []
        pred_y_samples.append(y_pred[i_max])
        true_y_samples.append(y_true[i_max])
        
        for npz in npz_files:
            res_npz = np.load(npz, allow_pickle=True)[sub_dataset][()]
            pred_y_samples.append(res_npz['y_pred'][i_max])
            true_y_samples.append(res_npz['y_true'][i_max])
            
        max_sample_len = np.max([len(i) for i in true_y_samples])
        y_true = true_y_samples[np.argwhere(max_sample_len == [len(i) for i in true_y_samples])[0][0]][::-1]
        x_true = np.linspace(0, max_sample_len, max_sample_len)
        
        plt.figure(figsize=(12,4))
        plt.plot(x_true, y_true, label='True label')
        colors = ['r','g', 'y']
        labels = ['LSTM with data from HI (RUL-ANN)', 'LSTM with data from deg. start', 'LSTM with data from op. start']
        
        x_samples = []
        for i in true_y_samples:
            x_samples.append(np.linspace(max_sample_len, max_sample_len-len(i), len(i)))
            
        rmse = []
        last_error = []
        for i in range(len(true_y_samples)):
            plt.plot(x_samples[i], pred_y_samples[i], colors[i], label= labels[i])
            rmse.append("{:.2f}".format(np.sqrt(np.mean(np.power(true_y_samples[i] - pred_y_samples[i][::-1], 2)))))
            last_error.append("{:.2f}".format(pred_y_samples[i][0] - true_y_samples[i][0][0]))
        plt.xlabel('Flight cycles')
        plt.ylabel('RUL')
        
        plt.text(0.65, 0.62, 'Last error LSTM with data from HI (RUL-ANN): '+last_error[0], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        plt.text(0.65, 0.54, 'Last error LSTM with data from deg. start: '+last_error[1], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)
        plt.text(0.65, 0.46, 'Last error LSTM with data from op. start: '+last_error[2], fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5), 
                 transform=plt.gca().transAxes)

        
        plt.legend()
        plt.savefig(os.path.join('images','LSTM_models_comparison_'+sub_dataset), bbox_inches='tight', pad_inches=0)       
model_sample_comparisson()