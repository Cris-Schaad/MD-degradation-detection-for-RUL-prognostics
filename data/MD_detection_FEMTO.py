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
time_step_per_image = data_set['time_step_per_measurement']
dataset_names = data_set['name']


time_window = 30
healthy_cycles = 6*30
    
x_data_healthy = np.asarray([i[:healthy_cycles] for i in x_data])    
m_d = MD.MahalanobisDistance(mode='scipy')
ms = m_d.fit_predict(np.concatenate(x_data_healthy))

x_train_md = np.asarray([m_d.fit(i) for i in x_data])
    
# plt.figure()
# for i in x_train_md:
#     plt.plot(i)
    
#Degradation detector
k = 3; n = 3
threshold = np.mean(ms) + 2*np.std(ms)
print('Threshold: {:.2f}'.format(threshold))
detector = dd.Detector(k, n, threshold = threshold)

# Training set
deg_ind = []
for sample in x_data:
    md_sample = m_d.fit(sample)
    test_deg = detector.detect(md_sample, prints=True)
        
    print('Test sample RUL: {:.0f}'.format(len(md_sample)-1-test_deg))
    deg_ind.append(test_deg)
    
    plt.figure()
    plt.plot(md_sample)
    plt.axhline(threshold, c='C1')
    plt.axvline(test_deg, c='r')
 

# Data to images
# del x_data; x_data = data_set['data_spectograms']      
# x_data, y_data = dp.time_window_sampling(x_data, y_data, time_step_per_image, 
#                                         time_step=time_step_per_image, 
#                                         add_last_dim=True, temporal_axis=1)

# x_data = dp.samples_reverse(x_data)
# y_data = dp.samples_reverse(y_data)

# # Sampling from degradation start index
# x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_ind, 0)

# Sampling from degradation start index
for i in range(len(x_data)):
    x_data[i] = np.column_stack((x_data[i], x_train_md[i]))
    
y_data = np.asarray([np.expand_dims(10*np.linspace(len(i),1, len(i)), axis=1) for i in x_data])
x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_ind, time_window) 
x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window, add_last_dim=False)

print(x_data[0].shape)   

# Saving
np.savez(os.path.join(data_dir, 'FEMTO_dataset.npz'),
        x_data = x_data,
        y_data = y_data,
        data_names = dataset_names,
        allow_pickle=True)
