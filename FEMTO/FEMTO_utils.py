import os
import numpy as np
from sklearn.model_selection import train_test_split


class FEMTO_importer():
    def __init__(self, dataset_name):
        
        npz = np.load(os.path.join('processed_data', dataset_name+'_dataset.npz'), allow_pickle=True)
        self.x_data = npz['x_data']
        self.y_data = npz['y_data']
        self.data_names = npz['data_names']
                
            
    def get_train_set(self, test_sample, valid_size=0.1):
        x_train = np.delete(self.x_data, test_sample)
        y_train = np.delete(self.y_data, test_sample)
        
        
        x_train, x_valid, y_train, y_valid  = train_test_split(np.concatenate(x_train), np.concatenate(y_train), 
                                                               test_size=valid_size,
                                                               random_state=0)
        return x_train, x_valid, y_train, y_valid 
        # x_train, x_valid, y_train, y_valid  = train_test_split(x_train, y_train, 
        #                                                        test_size=valid_size,
        #                                                        random_state=0)
        # return np.concatenate(x_train), np.concatenate(x_valid), np.concatenate(y_train), np.concatenate(y_valid)
    
    
    def get_train_samples(self, test_sample):
        x_train = np.delete(self.x_data, test_sample)
        y_train = np.delete(self.y_data, test_sample)
        return x_train, y_train

    def get_test_sample(self, test_sample):
        x_test = self.x_data[test_sample]        
        y_test = self.y_data[test_sample]             
        return x_test, y_test