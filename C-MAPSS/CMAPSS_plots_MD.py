import os
import numpy as np 
import matplotlib.pyplot as plt
from utils.data_processing import MinMaxScaler


plt.close('all')
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
RAW_NPZ = dict(np.load(os.path.join('processed_data', 'CMAPSS_raw.npz'), allow_pickle=True))
MD_NPZ = dict(np.load(os.path.join('processed_data', 'CMAPSS_MD_dataset.npz'), allow_pickle=True))


def plot_lifetime_distribution():
    
    for dataset in DATASETS:
        train_lens = []
        test_lens = []
        
        data = RAW_NPZ[dataset][()]
        for i in data['y_train']:
            train_lens.append(len(i))
    
        for i in data['y_test']:
            test_lens.append(len(i)+int(i[-1]))

        print(dataset)
        print('Train set mean lifetime', "{:.2f}".format(np.mean(train_lens)))            
        print('Test set mean lifetime', "{:.2f}".format(np.mean(test_lens)))
                        
        plt.figure()
        n, bins, patches = plt.hist(train_lens, 20, density=False, facecolor='C0', alpha=0.5, label='Train set')
        n, bins, patches = plt.hist(test_lens, bins, density=False, facecolor='C1', alpha=0.5, label='Test set')    
        
        plt.xlabel('Total engine lifetime')
        plt.ylabel('Number of samples')
        plt.title(dataset)
        plt.xlim(10, 550)
        plt.text(0.55, 0.78, 'Train set mean lifetime '+'{:.2f}'.format(np.mean(train_lens)), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.text(0.55, 0.7, 'Test set mean lifetime '+'{:.2f}'.format(np.mean(test_lens)), 
                 fontsize=10, bbox=dict(facecolor='green', alpha=0.3), transform=plt.gca().transAxes)
        plt.legend()
        plt.savefig(os.path.join('plots', 'life_dist_'+dataset), bbox_inches='tight', pad_inches=0)
# plot_lifetime_distribution()



def plot_md_DATASETS():

    for dataset in DATASETS:
        for set_name in ['train', 'test']:
            data = MD_NPZ[dataset][()]
            x_md = data['x_'+set_name+'_md']
            threshold = data['threshold']
            x_deg_start = data[set_name+'_deg']
    
            plt.figure(figsize=(9,3))
            for i, sample in enumerate(x_md):
                deg_ind = x_deg_start[i]
                no_deg_sample = sample[:deg_ind]
                deg_sample = sample[deg_ind:]
                plt.scatter(range(len(no_deg_sample)), no_deg_sample, color='g', alpha=0.5, linewidth=0.5, s=0.5)
                plt.scatter(range(len(no_deg_sample), len(no_deg_sample)+len(deg_sample)), 
                         deg_sample, color='r', alpha=0.5, linewidth=0.5, s=0.5)
                        
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
            plt.savefig(os.path.join('plots','MD_results',set_name+'_md_'+dataset), dpi=500,
                        bbox_inches='tight', pad_inches=0)
plot_md_DATASETS()


def plot_md_longest_shortest_samples_train():

    for dataset in DATASETS:
        data = MD_NPZ[dataset][()]
        x_md = data['x_train_md']
        threshold = data['threshold']
        x_deg_start = data['train_deg']
        
        x_lens = [len(x) for i, x in enumerate(x_md)]
        min_len_sample_ind = np.argwhere(x_lens == np.min(x_lens))[0][0]
        max_len_sample_ind = np.argwhere(x_lens == np.max(x_lens))[0][0]
        

        for ind, name in zip([min_len_sample_ind, max_len_sample_ind], ['min', 'max']):
            plt.figure(figsize=(8,4))
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

            plt.text(0.45, 0.9, 'Train set', fontsize=14, transform=plt.gca().transAxes)            
            plt.axhline(threshold, c='k')
            plt.xlabel('Operational cycles')
            plt.ylabel('Mahalanobis distance')
            plt.savefig(os.path.join('plots', 'MD_results','train_'+name+'_len_md_'+dataset), dpi=500,
                        bbox_inches='tight', pad_inches=0)
plot_md_longest_shortest_samples_train()


def plot_md_longest_deg_undeg_test():

    for dataset in DATASETS:
        data = MD_NPZ[dataset][()]
        x_md = data['x_test_md']
        threshold = data['threshold']
        x_deg_start = data['test_deg']
 
        for state in ['degraded', 'undegraded']:
            if state == 'undegraded':
                x_lens = [len(x) if x_deg_start[i] == len(x)-1 else 0 for i, x in enumerate(x_md)]
                max_len_ind = np.argwhere(x_lens == np.max(x_lens))[0][0]
            if state == 'degraded':
                x_lens = [len(x) if x_deg_start[i] < len(x)-1 else 0 for i, x in enumerate(x_md)]
                max_len_ind = np.argwhere(x_lens == np.max(x_lens))[0][0]
    
            plt.figure(figsize=(8,4))
            deg_ind = x_deg_start[max_len_ind]
            sample = x_md[max_len_ind]
            
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
    
            plt.text(0.42, 0.9, dataset + ' test set', fontsize=12, transform=plt.gca().transAxes)            
            plt.axhline(threshold, c='k')
            plt.xlabel('Operational cycles')
            plt.ylabel('Mahalanobis distance')
            plt.savefig(os.path.join('plots', 'MD_results','test_longest_'+state+'_md_'+dataset), dpi=500,
                        bbox_inches='tight', pad_inches=0)
plot_md_longest_deg_undeg_test()


def plot_ms_iters():

    plt.figure()
    for dataset in DATASETS:
        data = MD_NPZ[dataset][()]
        ms_iter = data['ms_iter']/len(np.concatenate(data['x_train_md']))
        plt.plot(ms_iter, label=dataset)
    plt.legend()
    plt.xlim((0))
    plt.ylim((0,1.1))
    plt.grid(dashes=(1,1))
    plt.xlabel('Number of iterations')
    plt.ylabel('MS proportion with respect to dataset')        
    plt.savefig(os.path.join('plots','MD_results','MS_prop'), bbox_inches='tight', pad_inches=0)
plot_ms_iters()



def input_image():
    
    dataset_npz = dict(np.load(os.path.join('processed_data', 'CMAPSS_dataset.npz'), allow_pickle=True))   
    x_train = dataset_npz['FD001'][()]['x_train']
    
    scaler_x = MinMaxScaler(feature_range=(0,1), feature_axis=2)
    x_train = [scaler_x.fit_transform(i, verbose=False) for i in x_train]
    sample = x_train[0][0,:,:,0]   
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(sample, cmap='Greys')
    ax.invert_yaxis()

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join('plots', 'input_image'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# input_image()