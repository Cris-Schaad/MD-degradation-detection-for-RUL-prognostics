import os
import numpy as np
from sklearn.model_selection import train_test_split


class CMAPSS_importer():
    
    def __init__(self, dataset_dir):
        self.dataset = dict(np.load(dataset_dir, allow_pickle=True))['dataset'][()]
        self.subdatasets = self.dataset.keys()

    def get_train_set(self, sub_dataset_name, valid_size=0.1):
        if type(valid_size) == float:
            x_train, x_valid, y_train, y_valid  = train_test_split(self.dataset[sub_dataset_name]['x_train'], self.dataset[sub_dataset_name]['y_train'], 
                                                                   test_size=valid_size)
            return np.concatenate(x_train), np.concatenate(x_valid), np.concatenate(y_train), np.concatenate(y_valid)   
        else:
            return np.concatenate(self.dataset[sub_dataset_name]['x_train']), np.concatenate(self.dataset[sub_dataset_name]['y_train'])
    
    def get_all_train_sets(self, valid_size=0.1):
        
        all_x_train = []
        all_y_train = []
        for sub_dataset_name in ['FD001', 'FD002', 'FD003', 'FD004']:
            all_x_train.append(self.dataset[sub_dataset_name]['x_train'])
            all_y_train.append(self.dataset[sub_dataset_name]['y_train'])
            
        x_train = np.concatenate(all_x_train)
        y_train = np.concatenate(all_y_train)
        
        x_train, x_valid, y_train, y_valid  = train_test_split(x_train, y_train, test_size=valid_size)
        return np.concatenate(x_train), np.concatenate(x_valid), np.concatenate(y_train), np.concatenate(y_valid)

    def get_test_set(self, sub_dataset_name):
        x_test = np.concatenate(self.dataset[sub_dataset_name]['x_test'])
        y_test = np.concatenate(self.dataset[sub_dataset_name]['y_test'])
        return x_test, y_test
    
    def get_test_set_for_metrics(self, sub_dataset_name, rul_end_index):
        x_test = self.dataset[sub_dataset_name]['x_test']
        y_test = self.dataset[sub_dataset_name]['y_test']
        
        x_test = np.asarray([sample[rul_end_index] for sample in x_test])
        y_test = np.asarray([sample[rul_end_index] for sample in y_test])
        return x_test, y_test 
    
    def get_train_samples(self, sub_dataset_name):
        return self.dataset[sub_dataset_name]['x_train'], self.dataset[sub_dataset_name]['y_train']
 
    def get_test_samples(self, sub_dataset_name):
        return self.dataset[sub_dataset_name]['x_test'], self.dataset[sub_dataset_name]['y_test']
    

    
class FEMTO_importer():
    def __init__(self, dataset_dir):
        
        npz = np.load(os.path.join(dataset_dir), allow_pickle=True)
        self.x_data = npz['x_data']
        self.y_data = npz['y_data']
        self.data_names = npz['data_names']
                
            
    def get_train_set(self, test_sample, valid_size=0.1):
        x_train = np.delete(self.x_data, test_sample)
        y_train = np.delete(self.y_data, test_sample)
  
        x_train, x_valid, y_train, y_valid  = train_test_split(x_train, y_train, test_size=valid_size)
        return np.concatenate(x_train), np.concatenate(x_valid), np.concatenate(y_train), np.concatenate(y_valid)
    
    
    def get_train_samples(self, test_sample):
        x_train = np.delete(self.x_data, test_sample)
        y_train = np.delete(self.y_data, test_sample)
        return x_train, y_train

    def get_test_sample(self, test_sample):
        x_test = self.x_data[test_sample]        
        y_test = self.y_data[test_sample]             
        return x_test, y_test