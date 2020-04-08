import os
import numpy as np
import matplotlib.pyplot as plt


import MahalanobisDistance as MD
import degradation_detection as dd
import data_processing as dp

plt.close('all')


data_dir = os.path.join('FEMTO', 'processed_data')
data_set = np.load(os.path.join(data_dir, 'FEMTO_processed_samples.npz'),  allow_pickle=True)

x_data = data_set['data_spectograms']
y_data = data_set['data_y']
time_step_per_image = data_set['time_step_per_measurement']
dataset_names = data_set['name']

time_window = 30
healthy_cycles = 180

for i, sample in enumerate(x_data):
    print('Analyzing {} as test sample'.format(dataset_names[i]))
    test_sample = data_set['data_features'][i]
    train_data = np.delete(data_set['data_features'], i)
    
    x_data_healthy = np.asarray([i[:healthy_cycles] for i in train_data])
    x_data_healthy = np.asarray([i for i in x_data_healthy])
    
    m_d = MD.MahalanobisDistance(mode='inv')
    m_d.fit_predict(np.concatenate(x_data_healthy))
    
    healthy_md = np.asarray([m_d.fit(i) for i in train_data])
        
    plt.figure()
    ini_md = []; end_md = []
    i_max = 0; min_dif = np.inf
    for i in healthy_md:
        if dd.cum_mean(i)[-1] < min_dif:
            i_max = i; min_dif = dd.cum_mean(i)[-1] 
        plt.plot(dd.cum_mean(i))
        

#x_data, y_data = dp.time_window_sampling(x_data, y_data, time_step_per_image, 
#                                         time_step=time_step_per_image, 
#                                         add_last_dim=True, temporal_axis=1)
#
#x_data = dp.samples_reverse(x_data)
#y_data = dp.samples_reverse(y_data)
#
#np.savez(os.path.join(data_dir, 'FEMTO_spectograms_RUL_dataset.npz'),
#         x_data = x_data,
#         y_data = y_data,
#         data_names = dataset_names,
#         allow_pickle=True)
