import os
import pandas as pd
import numpy as np
from scipy.signal import stft


import FEMTO_aux_functions as aux


data_samples = np.load(os.path.join('raw_data', 'FEMTO_raw_samples.npz'), allow_pickle=True)
data_raw = data_samples['data_raw']
sample_name = data_samples['name']


sample_spectogram = []
sample_features = []
sample_rul = []
for i in range(len(data_raw)):
    sample = data_raw[i]
    print(sample_name[i])       
    
    #Spectograms
    sampling = 25600        #hertz            
    win_len = 10            #miliseconds
    win_len = int(win_len*sampling//1000)
    win_len_time_step = 2560//win_len
    spectogram_pixel_overlap = 0

    subsamples = aux.windows(np.abs(sample), win_len, win_len)
    subsample_stft = stft(sample, nperseg=win_len, noverlap=spectogram_pixel_overlap, fs=sampling)
    
    subsamples_spectograms = np.absolute(subsample_stft[2])
    sample_spectogram.append(np.transpose(subsamples_spectograms))    
    
    #Features
    feature_win_len = 100             #miliseconds
    feature_win_len = int(feature_win_len*sampling//1000)
    
    subsamples = aux.windows(sample, feature_win_len, feature_win_len)        
    subsamples_features = np.asarray([aux.super_features(i) for i in subsamples])
    sample_features.append(subsamples_features)

    #RUL
    rul = subsample_stft[1]*1000
    sample_rul.append(np.expand_dims(rul[::-1], axis=-1))


sample_spectogram = np.asarray(sample_spectogram)        
for i in range(len(sample_spectogram)):
    sample_spectogram[i] = np.transpose(sample_spectogram[i])


np.savez(os.path.join('processed_data','FEMTO_processed_samples.npz'),
         data_features = sample_features,
         data_spectograms = sample_spectogram,
         data_rul = sample_rul,
         time_step_per_measurement = win_len_time_step,
         name = sample_name)