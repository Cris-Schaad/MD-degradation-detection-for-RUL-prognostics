import os
import numpy as np
import pandas as pd


folder = 'raw_data'
data_samples = os.listdir(folder)
data_samples.sort()


datasets = []
for sample in data_samples:
    datasets.append(sample)

raw_samples = []
sample_name = []
samples_rul = []
for sample in datasets:
    print('Processing: ', sample)
    data_files = os.listdir(os.path.join(folder, sample))
    vib_files = np.asarray([i for i in data_files if 'acc' in i])
    vib_files.sort()
    
    sample_data = []    
    for file in vib_files:
        dataframe = pd.read_csv(os.path.join(folder, sample, file), header=None)
        data_segment = np.asarray(dataframe.iloc[:,4])
        
        # if np.max(np.abs(data_segment)) >= 20:
        #     if np.max(np.abs(sample_data[-1])) >= 20:
        #         if np.max(np.abs(sample_data[-2])) >= 20:
        #             break
        sample_data.append(data_segment)
        
    sample_times = np.asarray([i/25600 for i in range(2560)])
    times = np.asarray([10*i for i in range(len(sample_data))])
    rul = np.asarray([sample_times+i for i in times]).flatten()[::-1]
    
    raw_samples.append(np.concatenate(sample_data))
    samples_rul.append(rul)
    sample_name.append(sample)
        
np.savez(os.path.join('processed_data', 'FEMTO_raw_samples.npz'),
         data_raw = raw_samples,
         data_rul = samples_rul,
         name = sample_name)            
