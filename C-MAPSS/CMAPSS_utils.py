import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def close_all():
    plt.close('all')
        
    
def get_data(x, op_cols, drop_cols):
    x_data = []
    x_op_settings = []
    for i in np.unique(x['Unit Number']):
        sample = x[x['Unit Number']==i] 
        sample_op_settings = sample[op_cols]
        sample = sample.drop(drop_cols, axis=1)
        
        x_data.append(np.asarray(sample))
        x_op_settings.append(np.asarray(sample_op_settings))
    return np.asarray(x_data),  np.asarray(x_op_settings)    

   
def rul(x, init, R_early):
   
    y = []
    for n in range(x.shape[0]):
        sample = x[n]
        if type(init)==int:
            rul_sample = np.asarray([i+1 for i in range(len(sample))])
            rul_sample = np.expand_dims(rul_sample[::-1], axis=-1)  
        else:      
            rul_sample = np.asarray([init[n]+i for i in range(len(sample))])
            rul_sample = rul_sample[::-1]  
        rul_sample = np.where(rul_sample >= R_early, R_early, rul_sample)
        y.append(rul_sample)     
    return x, np.asarray(y)



def load_CMAPSS_dataset(dataset, path):
    dataset_raw_npz = dict(np.load(os.path.join(path,'CMAPSS_raw.npz'), allow_pickle=True))
    dataset_dict = dataset_raw_npz[dataset][()]
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']
    return x_train, y_train, x_test, y_test



def load_CMAPSS_MD_dataset(dataset, path):
    dataset_raw_npz = dict(np.load(os.path.join(path,'CMAPSS_md_dataset.npz'), allow_pickle=True))
    dataset_dict = dataset_raw_npz[dataset][()]
    x_train_md = dataset_dict['x_train_md']
    x_test_md = dataset_dict['x_test_md']
    threshold = dataset_dict['threshold']
    return x_train_md, x_test_md, threshold



class CMAPSS_importer():
    def __init__(self, dataset_name):
        dataset_dir = os.path.join('processed_data', dataset_name+'_dataset.npz')
        self.dataset = np.load(dataset_dir, allow_pickle=True)
        self.subdatasets = self.dataset.keys()

    def get_train_set(self, sub_dataset_name, valid_size=0.1):
        sub_dataset = self.dataset[sub_dataset_name][()]        
        if type(valid_size) == float:
            x_train, x_valid, y_train, y_valid = train_test_split(sub_dataset['x_train'], 
                                                                  sub_dataset['y_train'], 
                                                                  test_size=valid_size,
                                                                  random_state=0)
            return np.concatenate(x_train), np.concatenate(x_valid), np.concatenate(y_train), np.concatenate(y_valid)   
        else:
            return np.concatenate(sub_dataset['x_train']), np.concatenate(sub_dataset['y_train'])
    
    def get_test_set(self, sub_dataset_name):
        x_test = np.concatenate(self.dataset[sub_dataset_name][()]['x_test'])
        y_test = np.concatenate(self.dataset[sub_dataset_name][()]['y_test'])
        return x_test, y_test
    
    def get_test_set_for_metrics(self, sub_dataset_name, rul_end_index):
        x_test = self.dataset[sub_dataset_name][()]['x_test']
        y_test = self.dataset[sub_dataset_name][()]['y_test']
        
        x_test = np.asarray([sample[rul_end_index] for sample in x_test])
        y_test = np.asarray([sample[rul_end_index] for sample in y_test])
        return x_test, y_test 
    
    def get_samples(self, sub_dataset_name, set_name):
        return self.dataset[sub_dataset_name][()]['x_'+set_name], self.dataset[sub_dataset_name][()]['y_'+set_name]
 
