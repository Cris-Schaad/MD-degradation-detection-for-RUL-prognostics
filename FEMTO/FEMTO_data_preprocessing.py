import os
import sys 

import numpy as np
import pandas as pd
from scipy.signal import stft

sys.path.append('..')
from utils.data_processing import windows
from utils.data_processing import features


def raw_to_npz():
    folder = 'raw_data'
    data_samples = os.listdir(folder)
    data_samples.sort()
    
    raw_samples = []
    sample_name = []
    for sample in data_samples:
        print('Processing: ', sample)
        data_files = os.listdir(os.path.join(folder, sample))
        vib_files = np.asarray([i for i in data_files if 'acc' in i])
        vib_files.sort()
        
        sample_data = []   
        ignore_files = 0# 6*30
        for file in vib_files[ignore_files:]:
            dataframe = pd.read_csv(os.path.join(folder, sample, file), header=None)
            data_segment = np.asarray(dataframe.iloc[:,4])

            if np.max(np.abs(data_segment)) >= 20:
                if np.max(np.abs(sample_data[-1])) >= 20:
                    if np.max(np.abs(sample_data[-2])) >= 20:
                        break
                    
            sample_data.append(data_segment)
        raw_samples.append(np.concatenate(sample_data))
        sample_name.append(sample)
            
    np.savez(os.path.join('processed_data', 'FEMTO_raw_samples.npz'),
             data_raw = raw_samples,
             name = sample_name)            
raw_to_npz()


def data_to_spectograms():
    data_samples = np.load(os.path.join('processed_data', 'FEMTO_raw_samples.npz'), allow_pickle=True)
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
        win_len = 2.5            #miliseconds
        win_len = int(win_len*sampling//1000)
        win_len_time_step = 2560//win_len
    
        subsample_stft = stft(sample, nperseg=win_len, noverlap=0, fs=sampling)
        subsamples_spectograms = np.absolute(subsample_stft[2])
        sample_spectogram.append(np.transpose(subsamples_spectograms))   
        
        #Features
        feature_win_len = 100             #miliseconds
        feature_win_len = int(feature_win_len*sampling//1000)
        feature_win_len_time_step = 2560//feature_win_len

        subsamples = windows(sample, feature_win_len, feature_win_len)        
        subsamples_features = np.asarray([features(i) for i in subsamples])
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
             spec_timestep = win_len_time_step,
             feature_timestep = feature_win_len_time_step,
             name = sample_name)
data_to_spectograms()