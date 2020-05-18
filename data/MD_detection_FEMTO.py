import os
import numpy as np
import matplotlib.pyplot as plt

import MS_iterator as MS_iterator
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

time_window = 6

#Degradation detector
k = 10; n = 12; sigma = 1

iterator = MS_iterator.iterator(k, n, sigma)
deg_start_ind, threshold = iterator.iterative_calculation(x_data, verbose=False)
x_data_md = iterator.md_calculation_op(x_data)

# Training set
end_md = []
for i, sample in enumerate(x_data_md):
    test_deg = deg_start_ind[i]
        
    print(dataset_names[i]+' sample RUL: {:.0f}'.format(len(sample)-test_deg))
    end_md.append(sample[-1])
    
    plt.figure()
    plt.plot(sample)
    plt.axhline(threshold, c='C1')
    plt.axvline(test_deg, c='r')
    plt.title(dataset_names[i])

plt.figure()
plt.hist(end_md, 10)
 

# Data to images
del x_data
x_data, y_data = dp.time_window_sampling(x_spec, y_data, timestep_per_image, 
                                         temporal_axis=1,
                                         time_step=timestep_per_image,
                                         add_last_dim=True)
x_data = dp.samples_reverse(x_data)
    
    
# Sampling from degradation start index
detector = iterator.get_detector()
deg_start_ind = [i//2 for i in deg_start_ind]

y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])
x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_start_ind, time_window)

# Sequence of images
x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window)
print(x_data.shape)

# # Data sequences
# y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])

# # Sampling from degradation start index
# detector = iterator.get_detector()
# x_data, y_data = detector.sampling_from_index(x_data, y_data, deg_start_ind, time_window)  
# x_data, y_data = dp.time_window_sampling(x_data, y_data, time_window)


# Saving
np.savez(os.path.join(data_dir, 'FEMTO_dataset.npz'),
        x_data = x_data,
        y_data = y_data,
        data_names = dataset_names)
