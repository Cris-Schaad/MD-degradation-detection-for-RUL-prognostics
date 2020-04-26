import os
import numpy as np
import matplotlib.pyplot as plt


import MahalanobisDistance as MD
import degradation_detection as dd
import data_processing as dp

plt.close('all')


data_dir = os.path.join('FEMTO', 'processed_data')
data_set = np.load(os.path.join(data_dir, 'FEMTO_processed_samples.npz'),  allow_pickle=True)

x_data = np.delete(data_set['data_features'], [9,11])
x_spec = np.delete(data_set['data_spectograms'], [9,11])
y_data = np.delete(data_set['data_rul'], [9,11])
timestep_per_image = data_set['spec_timestep']
feature_timestep_per_image = data_set['feature_timestep']
dataset_names = np.delete(data_set['name'], [9,11])

# x_data = data_set['data_features'] 
# x_spec = data_set['data_spectograms']
# y_data = data_set['data_rul'] 
# timestep_per_image = data_set['spec_timestep']
# feature_timestep_per_image = data_set['feature_timestep']
# dataset_names = data_set['name'] 

time_window = 12
healthy_cycles = 6*30
    
x_data_healthy = np.asarray([i[:healthy_cycles] for i in x_data])    
m_d = MD.MahalanobisDistance(mode='covariance')
m_d.fit_predict(np.concatenate(x_data_healthy))

x_data_md = np.asarray([m_d.fit(i) for i in x_data])
# x_data_md = np.asarray([i/i[0] for i in x_data_md])
ms = np.asarray([i[:healthy_cycles] for i in x_data_md])
print('Mean MD in MS: {:.2f}'.format(np.mean(ms)))


#Degradation detector
k = 5; n = 6
threshold = np.mean(ms) + 1*np.std(ms)
print('Threshold: {:.2f}'.format(threshold))
detector = dd.Detector(k, n, threshold = threshold)

# Training set
deg_ind = []; end_md = []
for i, sample in enumerate(x_data):
    md_sample = m_d.fit(sample)
    test_deg = detector.detect(md_sample, prints=True)
        
    print('Test sample RUL: {:.0f}'.format(len(md_sample)-1-test_deg))
    deg_ind.append(test_deg)
    end_md.append(md_sample[-1])
    
    plt.figure()
    plt.plot(md_sample)
    plt.axhline(threshold, c='C1')
    plt.axvline(test_deg, c='r')
    plt.title(dataset_names[i])

plt.figure()
plt.hist(end_md, 10)
 

# Data to images
x_data, y_data = dp.time_window_sampling(x_spec, y_data, timestep_per_image, 
                                        time_step=timestep_per_image, 
                                        add_last_dim=True, temporal_axis=1)
x_data = dp.samples_reverse(x_data)
    
    
# Sampling from degradation start index
y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])
x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_ind, time_window-1)

# Sequence of images
x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window)
# 

# # Data sequences
# y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])

# # Sampling from degradation start index
# x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_ind, time_window-1)  
# x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window)


# Saving
np.savez(os.path.join(data_dir, 'FEMTO_dataset.npz'),
        x_data = x_data,
        y_data = y_data,
        data_names = dataset_names,
        allow_pickle=True)
