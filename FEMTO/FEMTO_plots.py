import os

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')


def plot_raw_samples():
    dataset_raw_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_raw_samples.npz'), allow_pickle=True))    
    samples_name = dataset_raw_npz['name']
    samples_raw = dataset_raw_npz['data_raw']
    
    
    for name, data in zip(samples_name, samples_raw):      
       
        plt.figure()
        plt.plot(range(len(data)), data)
        plt.title(name)
# plot_raw_samples()   
  

def plot_md_datasets():
    dataset_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_dataset.npz'), allow_pickle=True))    
    x_md = dataset_npz['x_data_md']
    x_deg_start = dataset_npz['x_deg_start']
    samples_name = dataset_npz['data_names']
   
    plt.close('all')
    fig, axs = plt.subplots(5,3, figsize=(10,15), constrained_layout=True)
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
        if i in [0,3,6,9,12]:
            axs[i].set_ylabel('Mahalanobis distance')
        if i in [1,4,7,10,13]:
            axs[i].set_xlabel('Feature measurement N°')
    
    fig.savefig(os.path.join('plots', 'MD_dataset'), dpi=500,
                bbox_inches='tight', pad_inches=0)
plot_md_datasets()


def spectrograms():
    dataset_npz = dict(np.load(os.path.join('processed_data', 'FEMTO_processed_samples.npz'), allow_pickle=True))    
    specs = dataset_npz['data_spectograms']
    spec_timestep = dataset_npz['spec_timestep']

    sample = specs[0]
    sample = sample[:, -spec_timestep*6:]
    
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(sample)
    ax.invert_yaxis()

    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join('plots', 'spectogram'), dpi=500,
                bbox_inches='tight', pad_inches=0)
# spectrograms()

#dataset = np.load('FEMTO_raw_samples.npz', allow_pickle=True)
#raw_data = dataset['data_raw']
#rul_data = dataset['data_rul']
#
#sample = 0
#
#sample_data = raw_data[sample]
#sample_rul = rul_data[sample]
#
#plt.figure(figsize=(12,4))
#plt.plot(sample_rul[::-1]/3600, sample_data)
#plt.xlabel('Time [hours]')
#plt.ylabel('Vertical acceleration [g]')
#plt.grid()
#plt.savefig(os.path.join('images', 'vertical_vib_sample'+str(sample)))

#
##DBSCAN parameters
#cluster_radius = 0.01
#min_cluster_points = 1000
#
##Degradation detection parameters
#n_conditions = 4
#conditions_time_window = 6
#
#data_set = np.load('FEMTO_processed_samples.npz',  allow_pickle=True)
#dbscan_cluster = fun.DBSCAN_degradation_start(data_set['data_features'])
#dbscan_cluster.fit_predict(cluster_radius, min_cluster_points)
#
#indx_start = dbscan_cluster.degradation_start(n_conditions, conditions_time_window)
#dbscan_cluster.cluster_plot(save_fig=True)
#dbscan_cluster.degradation_start_per_sample()
#
#for i in range(len(data_set['data_features'])):
#    dbscan_cluster.sample_plot(i, 1, 'Maximum acceleration', save_fig=True)


# def image_zoom(array, y_factor, x_factor):
#     y_dim, x_dim = array.shape
#     zoomed_array = np.zeros((int(y_factor*y_dim), int(x_factor*x_dim)))
    
#     for i in range(y_dim):
#         for j in range(x_dim):
#             i_ind_1 = i*y_factor
#             i_ind_2 = (i+1)*y_factor   
#             j_ind_1 = j*x_factor
#             j_ind_2 = (j+1)*x_factor    
#             zoomed_array[i_ind_1:i_ind_2, j_ind_1:j_ind_2] = array[i,j]
            
#     return zoomed_array
    
 
# dataset = np.load(os.path.join('processed_data','FEMTO_dataset.npz'), allow_pickle=True)
# x_data = dataset['x_data']


# scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=1)
# scaler_x.fit_transform(np.concatenate(x_data))

# for i in range(1):
#     print(i)
#     sample_spec = x_data[i] #scaler_x.transform(x_data[i])
#     t,n,h,w,c = sample_spec.shape
#     super_spec = np.concatenate(sample_spec[:,-1], axis=1)[:,:,0]
#     print(super_spec.shape)
#     zoomed_spec = image_zoom(super_spec, 10, 10)
#     plt.figure(figsize=(10,2))
#     plt.contourf(zoomed_spec, levels=100, cmap='inferno')
#     cbar = plt.colorbar(format='%.2f')
#     cbar.ax.set_ylabel('Frequency intensity')
#     plt.savefig(os.path.join('images','spectogram_'+str(i)), bbox_inches='tight', pad_inches=0)

#
#sample_spec = sample_spec[-1][:,:,0]
#zoomed_spec = image_zoom(sample_spec, 10, 10)
#
#plt.figure()
#plt.contourf(zoomed_spec, levels=100)