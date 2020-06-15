import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from MSIterativeAlgorithm import MSIterativeAlgorithm
from utils.data_processing import time_window_sampling
from utils.data_processing import samples_reverse


plt.close('all')
data_set = np.load(os.path.join('processed_data', 'FEMTO_processed_samples.npz'),  allow_pickle=True)

x_data = np.delete(data_set['data_features'], [9,11])
x_spec = np.delete(data_set['data_spectograms'], [9,11])
y_data = np.delete(data_set['data_rul'], [9,11])
timestep_per_image = data_set['spec_timestep']
dataset_names = np.delete(data_set['name'], [9,11])

time_window = 12

#Degradation detector
k = 6; n = 6; sigma = 4

iterator = MSIterativeAlgorithm(k, n, sigma)
iterator.iterative_calculation(x_data, verbose=False)

x_data_md = iterator.md_calculation_op(x_data)
deg_start_ind = iterator.detect_degradation_start(x_data, False)
threshold = iterator.threshold

for i, sample in enumerate(x_data_md):
    test_deg = deg_start_ind[i]
        
    print(dataset_names[i]+' sample RUL: {:.0f}'.format(len(sample)-test_deg))    
    plt.figure()
    plt.plot(sample)
    plt.axhline(threshold, c='C1')
    plt.axvline(test_deg, c='r')
    plt.title(dataset_names[i])


# # Data to images
# x_data, y_data = time_window_sampling(x_spec, y_data, timestep_per_image,temporal_axis=1,
#                                       time_step=timestep_per_image, add_last_dim=True)
# x_data = samples_reverse(x_data)

# # Sampling from degradation start index
# deg_start_ind = [i//data_set['feature_timestep'] for i in deg_start_ind]
# y_data = np.asarray([np.expand_dims(10*np.linspace(len(i), 1, len(i)), axis=1) for i in x_data])
# x_data, y_data = iterator.detector.sampling_from_index(x_data, y_data, deg_start_ind, time_window)


    
# # Sequence of images
# x_data, y_data = time_window_sampling(x_data, y_data, time_window)


# # Saving
# np.savez(os.path.join('processed_data', 'FEMTO_dataset.npz'),
#         x_data = x_data,
#         y_data = y_data,
#         data_names = dataset_names)
