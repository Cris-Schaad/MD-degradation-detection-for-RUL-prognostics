import os
import numpy as np
import matplotlib.pyplot as plt


import MahalanobisDistance as MD
import degradation_detection as dd
import data_processing as dp

plt.close('all')


data_dir = os.path.join('FEMTO', 'processed_data')
data_set = np.load(os.path.join(data_dir, 'FEMTO_processed_samples.npz'),  allow_pickle=True)

x_data = data_set['data_features']
y_data = data_set['data_rul']
timestep_per_image = data_set['spec_timestep']
feature_timestep_per_image = data_set['feature_timestep']
dataset_names = data_set['name']


def cum_mean(x, time_window):
    return np.asarray([np.mean(x[:i]) for i in range(time_window, len(x))])


time_window = 6
healthy_cycles = 6*30
    
x_data_healthy = np.asarray([i[:healthy_cycles] for i in x_data])    
m_d = MD.MahalanobisDistance(mode='covariance')
ms = m_d.fit_predict(np.concatenate(x_data_healthy))
print('Mean MD in MS: {:.2f}'.format(np.mean(ms)))

x_train_md = np.asarray([m_d.fit(i) for i in x_data])
    
# plt.figure()
# for i in x_train_md:
#     plt.plot(i)
    
#Degradation detector
k = 2; n = 3
threshold = np.mean(ms) + 2*np.std(ms)
detector = dd.Detector(k, n, threshold = threshold)

# Training set
deg_ind = []; end_md = []
for sample in x_data:
    # md_sample = cum_mean(m_d.fit(sample), 6*feature_timestep_per_image)
    md_sample = m_d.fit(sample)
    test_deg = detector.detect(md_sample, prints=True)
        
    print('Test sample RUL: {:.0f}'.format(len(md_sample)-1-test_deg))
    deg_ind.append(test_deg/feature_timestep_per_image)
    end_md.append(md_sample[-1])
    
    plt.figure()
    plt.plot(md_sample)
    plt.axhline(threshold, c='C1')
    plt.axvline(test_deg, c='r')
    
plt.figure()
plt.hist(end_md, 10)
 

# # Data to images
# del x_data; x_data = data_set['data_spectograms']  
# x_data, y_data = dp.time_window_sampling(x_data, y_data, timestep_per_image, 
#                                         time_step=timestep_per_image, 
#                                         add_last_dim=True, temporal_axis=1)
# x_data = dp.samples_reverse(x_data)


# # Sampling from degradation start index
# y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])
# x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_ind, time_window)


# # Sequence of images
# x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window)


# # Saving
# np.savez(os.path.join(data_dir, 'FEMTO_dataset.npz'),
#         x_data = x_data,
#         y_data = y_data,
#         data_names = dataset_names,
#         allow_pickle=True)
