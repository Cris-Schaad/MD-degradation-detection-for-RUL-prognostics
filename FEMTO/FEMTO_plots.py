import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.close('all')
dataset = 'FEMTO'
model = 'ConvLSTM'

def RUL_RMSE_stats():
    results_dir = os.path.join(os.getcwd(), 'training_results', dataset, model)
    samples = [i for i in os.listdir(results_dir) if 'Bearing' in i ]
    for sub_dataset in samples:
        
        data = pd.read_csv(os.path.join(results_dir, sub_dataset, sub_dataset+'.csv'), sep=',')
        print(sub_dataset+' min RMSE: {:.2f}'.format(np.min(data[' RMSE'])))
        print(sub_dataset+' mean RMSE: {:.2f}'.format(np.mean(data[' RMSE'])))
        print(sub_dataset+' std. RMSE: {:.2f}\n'.format(np.std(data[' RMSE'])))        
# RUL_RMSE_stats()  


def plot_raw_samples():
    dataset_raw_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_raw_samples.npz'), allow_pickle=True))    
    samples_name = dataset_raw_npz['name']
    samples_raw = dataset_raw_npz['data_raw']
    
    
    for name, data in zip(samples_name, samples_raw):   

        plt.figure(figsize=(8,4.5))
        for i in range(int(len(data)/2560)):
            x = np.linspace(10*i, 10*i+0.1, 2560)/3600
            plt.plot(x, data[i*2560:(i+1)*2560], c='C0', linewidth=0.5)
        plt.xlabel('Operation time [hours]')
        plt.ylabel('Acceleration [g]')
        plt.title(name)
plot_raw_samples()   



def plot_sample_len_dist():
    dataset_raw_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_raw_samples.npz'), allow_pickle=True))    
    lens = [len(i)/(360*2560) for i in dataset_raw_npz['data_raw']]
    
    plt.figure(figsize=(5,5))
    plt.hist(lens, bins=10)
    plt.xlabel('Operation time [hours]')
    plt.ylabel('Acceleration [g]')
    plt.title('FEMTO sample lifetimes distribution')
    plt.savefig(os.path.join('plots', 'lifetime_dist'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# plot_sample_len_dist()   
  

def plot_ms_iters():
    plt.close('all')
    plt.figure()
    dataset_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_dataset.npz'), allow_pickle=True))    
    ms_iter = dataset_npz['ms_iter']
    plt.plot(ms_iter/np.max(ms_iter))
    plt.xlim((0))
    plt.ylim((0,1.1))
    plt.grid(dashes=(1,1))
    plt.xlabel('Number of iterations')
    plt.ylabel('MS proportion with respect to dataset')        
    plt.savefig(os.path.join('plots','MS_prop'), bbox_inches='tight', pad_inches=0)
# plot_ms_iters()


def plot_md_datasets():
    dataset_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_dataset.npz'), allow_pickle=True))    
    x_md = dataset_npz['x_data_md']
    x_deg_start = dataset_npz['x_deg_start']
    samples_name = dataset_npz['data_names']
   
    plt.close('all')
    # fig, axs = plt.subplots(5,3, figsize=(10,15), constrained_layout=True)
    fig, axs = plt.subplots(3,5, figsize=(16,6), constrained_layout=True)
    axs = axs.flat
    for i, name in enumerate(samples_name):
        sample = x_md[i]
        deg_ind = x_deg_start[i]
        no_deg_sample = sample[:deg_ind]
        deg_sample = sample[deg_ind:]
        axs[i].plot(range(len(no_deg_sample)), no_deg_sample, color='g')
        axs[i].plot(range(len(no_deg_sample), len(no_deg_sample)+len(deg_sample)), 
                 deg_sample, color='r')
        # axs[i].set_title(name)
        axs[i].axvline(deg_ind, c='C0')
        axs[i].text(0.35, 0.85, name, 
                  fontsize=10, bbox=dict(facecolor='white', alpha=1, edgecolor='white'), transform=axs[i].transAxes)
        if i in [0,5,10]: #[0,3,6,9,12]:
            axs[i].set_ylabel('Mahalanobis distance')
        if i in [2,7,12]: #[1,4,7,10,13]:
            axs[i].set_xlabel('Feature measurement N°')
    
    fig.savefig(os.path.join('plots', 'MD_dataset'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# plot_md_datasets()



def plot_RUL_samples():
    results_dir = os.path.join(os.getcwd(), 'training_results', dataset, model)
    samples = [i for i in os.listdir(results_dir) if 'Bearing' in i ]
    
    plt.close('all')
    # fig, axs = plt.subplots(5,3, figsize=(10,15), constrained_layout=True)
    fig, axs = plt.subplots(3,5, figsize=(16,6), constrained_layout=True)
    axs = axs.flat
    for i, name in enumerate(samples):
        res_rmse = pd.read_csv(os.path.join(results_dir, name, name+'.csv'))
        best_model_name = 'model_'+str(res_rmse.iloc[res_rmse[' RMSE'].idxmin(),0])+'_results.npz'
        res_data = np.load(os.path.join(results_dir, name, best_model_name),allow_pickle=True)
        
        sample_true = res_data['y_true']
        sample_pred = res_data['y_pred']
        sample_rmse = np.sqrt(np.mean(np.power(sample_true - sample_pred, 2)))


        axs[i].plot(np.arange(1, len(sample_true)+1,1), sample_true[::-1], label='True RUL')    
        axs[i].plot(np.arange(1, len(sample_true)+1,1), sample_pred[::-1], 'r', label='Predicted RUL')
        if np.max(sample_true)> np.min(sample_pred):
            axs[i].set_ylim(0, 1.2*np.max(sample_true))
        else:
            axs[i].set_ylim(0, 1.2*np.max(sample_pred))
            
        # axs[i].set_title(name)
        # axs[i].text(0.65, 0.7, 'Sample RMSE '+'{:.2f}'.format(sample_rmse), 
        #          fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        
        axs[i].text(0.35, 0.85, name, 
                  fontsize=10, bbox=dict(facecolor='white', alpha=0, edgecolor='white'), transform=axs[i].transAxes)        
        if i in [0,5,10]: #[0,3,6,9,12]:
            axs[i].set_ylabel('RUL [s]')
        if i in [2,7,12]: #[1,4,7,10,13]:
            axs[i].set_xlabel('Operating time [s]')
    
    fig.savefig(os.path.join('plots', 'RUL_predictions'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# plot_RUL_samples()



def spectrograms():
    dataset_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_processed_samples.npz'), allow_pickle=True))    
    specs = dataset_npz['data_spectograms']
    spec_timestep = dataset_npz['spec_timestep']

    sample = specs[0]
    sample = sample[:, -spec_timestep*6:]
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(sample)
    ax.invert_yaxis()

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join('plots', 'spectogram'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# spectrograms()