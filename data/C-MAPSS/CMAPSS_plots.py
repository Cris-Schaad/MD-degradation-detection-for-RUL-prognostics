import os
import numpy as np 
import matplotlib.pyplot as plt


def plot_lifetime_distribution():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_raw_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_raw.npz'), allow_pickle=True))
    
    plt.close('all')
    
    for dataset in datasets:
        train_lens = []
        test_lens = []
        
        data = dataset_raw_npz[dataset][()]
        for i in data['y_train']:
            train_lens.append(len(i))
    
        for i in data['y_test']:
            test_lens.append(len(i)+int(i[-1]))

        print(dataset)
        print('Train set mean lifetime', "{:.2f}".format(np.mean(train_lens)))            
        print('Test set mean lifetime', "{:.2f}".format(np.mean(test_lens)))
                        
        plt.figure()
        n, bins, patches = plt.hist(train_lens, 20, density=False, facecolor='b', alpha=0.5, label='Train set')
        n, bins, patches = plt.hist(test_lens, bins, density=False, facecolor='r', alpha=0.5, label='Test set')    
        
        plt.xlabel('Total engine lifetime '+dataset)
        plt.ylabel('Number of samples')
        plt.xlim(10, 550)
        plt.grid(True)
        plt.text(0.55, 0.78, 'Train set mean lifetime '+'{:.2f}'.format(np.mean(train_lens)), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.text(0.55, 0.7, 'Test set mean lifetime '+'{:.2f}'.format(np.mean(test_lens)), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.legend()
        plt.savefig(os.path.join('plots','life_dist_'+dataset), bbox_inches='tight', pad_inches=0)
# plot_lifetime_distribution()



def plot_md_datasets():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True))
    
    plt.close('all')
    for dataset in datasets:
        for set_name in ['train', 'test']:
            data = dataset_npz[dataset][()]
            x_md = data['x_'+set_name+'_md']
            threshold = data['threshold']
            x_deg_start = data[set_name+'_deg']
    
            plt.figure(figsize=(9,3))
            for i, sample in enumerate(x_md):
                deg_ind = x_deg_start[i]
                no_deg_sample = sample[:deg_ind]
                deg_sample = sample[deg_ind:]
                plt.plot(range(len(no_deg_sample)), no_deg_sample, color='g', alpha=0.5, linewidth=0.5)
                plt.plot(range(len(no_deg_sample), len(no_deg_sample)+len(deg_sample)), 
                         deg_sample, color='r', alpha=0.5, linewidth=0.5)
                        
            plt.legend(('Healthy op.', 'Degradative op.'), loc='upper right', framealpha=1)
            ax = plt.gca()
            leg = ax.get_legend()
            leg.legendHandles[0].set_color('g')
            leg.legendHandles[0].set_alpha(1)
            leg.legendHandles[0].set_linewidth(2)
            leg.legendHandles[1].set_color('r')
            leg.legendHandles[1].set_alpha(1)
            leg.legendHandles[1].set_linewidth(2)
            
            plt.text(0.05, 0.85, 'Threshold MD: {:.2f}'.format(threshold), 
                     fontsize=10, bbox=dict(facecolor='yellow', alpha=0.3), transform=plt.gca().transAxes)
            
            plt.axhline(threshold, c='k')
            plt.xlabel('Operational cycles')
            plt.ylabel('Mahalanobis distance')
            plt.ylim((0,8))
            plt.savefig(os.path.join('plots',set_name+'_md_'+dataset), dpi=500,
                        bbox_inches='tight', pad_inches=0)
# plot_md_datasets()


def plot_md_samples():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True))
    
    plt.close('all')
    for dataset in datasets:
        data = dataset_npz[dataset][()]
        x_md = data['x_train_md']
        threshold = data['threshold']
        x_deg_start = data['train_deg']
        
        x_lens = [len(i) for i in x_md]
        min_len_sample_ind = np.argwhere(x_lens == np.min(x_lens))[0][0]
        max_len_sample_ind = np.argwhere(x_lens == np.max(x_lens))[0][0]
        

        for ind, name in zip([min_len_sample_ind, max_len_sample_ind], ['min', 'max']):
            plt.figure(figsize=(9,3))
            deg_ind = x_deg_start[ind]
            sample = x_md[ind]
            
            no_deg_sample = sample[:deg_ind]
            deg_sample = sample[deg_ind:]
            plt.plot(range(len(no_deg_sample)), no_deg_sample, color='g', alpha=0.5, linewidth=2)
            plt.plot(range(len(no_deg_sample), len(no_deg_sample)+len(deg_sample)), 
                     deg_sample, color='r', alpha=0.5, linewidth=2)
                    
            plt.legend(('Healthy op.', 'Degradative op.'), loc='upper right', framealpha=1)
            ax = plt.gca()
            leg = ax.get_legend()
            leg.legendHandles[0].set_color('g')
            leg.legendHandles[0].set_alpha(1)
            leg.legendHandles[0].set_linewidth(2)
            leg.legendHandles[1].set_color('r')
            leg.legendHandles[1].set_alpha(1)
            leg.legendHandles[1].set_linewidth(2)
            
            plt.text(0.05, 0.85, 'Threshold MD: {:.2f}'.format(threshold), 
                     fontsize=10, bbox=dict(facecolor='yellow', alpha=0.3), transform=plt.gca().transAxes)
            
            plt.axhline(threshold, c='k')
            plt.xlabel('Operational cycles')
            plt.ylabel('Mahalanobis distance')
            # plt.ylim((0,8))
            plt.savefig(os.path.join('plots', name+'len_md_'+dataset), dpi=500,
                        bbox_inches='tight', pad_inches=0)
plot_md_samples()


def plot_ms_iters():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True))['dataset'][()]
    
    plt.close('all')
    plt.figure()
    for dataset in datasets:
        data = dataset_npz[dataset]
        ms_iter = data['ms_iter']/len(np.concatenate(data['x_train_md']))
        plt.plot(ms_iter, label=dataset)
    plt.legend()
    plt.xlim((0))
    plt.ylim((0,1.1))
    plt.grid(dashes=(1,1))
    plt.xlabel('Number of iterations')
    plt.ylabel('MS proportion with respect to dataset')        
    plt.savefig(os.path.join('plots','MS_prop'), bbox_inches='tight', pad_inches=0)
# plot_ms_iters()


def plot_traing_set_size_change():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_md_dataset.npz'), allow_pickle=True))['dataset'][()]
    
    plt.close('all')
    for dataset in datasets:
        data = dataset_npz[dataset]
        x_md = data['x_train_md']
        x_deg_start = data['train_deg']

        
        original_training_set_ruls = []
        new_training_set_ruls = []
        for i, sample in enumerate(x_md):
            deg_ind = x_deg_start[i]
            original_training_set_ruls.append(np.arange(0, len(sample),1))
            new_training_set_ruls.append(np.arange(0, deg_ind+1, 1))
            
        original_training_set_ruls = np.concatenate(original_training_set_ruls)
        new_training_set_ruls = np.concatenate(new_training_set_ruls)
                        
        plt.figure(figsize=(9,3))    
        plt.hist(original_training_set_ruls, bins=100, range=(0,500), alpha=0.6, label='Training set RULs from op. start', color='r')
        plt.hist(new_training_set_ruls, bins=100, range=(0,500), alpha=0.7, label='Training set RULs from deg. start', color='g')

        plt.legend()
        plt.xlabel('Lifetime flight cycles')
        plt.ylabel('Number of samples')
        plt.savefig(os.path.join('plots', dataset+'_training_set_rul_dist'), dpi=500,
                    bbox_inches='tight', pad_inches=0)
# plot_traing_set_size_change()