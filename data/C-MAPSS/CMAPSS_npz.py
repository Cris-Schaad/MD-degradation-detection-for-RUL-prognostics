import os

import numpy as np
import pandas as pd



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


def load_from_txt(dataset, R_early = np.inf):
    
    if dataset == 'FD001' or 'FD003':
        n_flights = 100
       
    if dataset == 'FD002':
        n_flights = 259
        
    if dataset == 'FD004':
        n_flights = 248

    path = os.getcwd()
    x_train = pd.read_csv(open(os.path.join(path,'data','train_'+dataset+'.txt'),'r', encoding='utf8'), delim_whitespace=True, header=None)
    x_test = pd.read_csv(open(os.path.join(path,'data','test_'+dataset+'.txt'),'r', encoding='utf8'),delim_whitespace=True,header=None)    
    y_test = np.array(pd.read_csv(open(os.path.join(path,'data','RUL_'+dataset+'.txt'),'r', encoding='utf8'), delim_whitespace=True,header=None))
    
    column_names=['Unit Number','Cycles', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',
                    'Sensor Measurement  1', 'Sensor Measurement  2','Sensor Measurement  3','Sensor Measurement  4',
                    'Sensor Measurement  5','Sensor Measurement  6','Sensor Measurement  7','Sensor Measurement  8',
                    'Sensor Measurement  9','Sensor Measurement  10','Sensor Measurement  11','Sensor Measurement  12',
                    'Sensor Measurement  13','Sensor Measurement  14','Sensor Measurement  15','Sensor Measurement  16',
                    'Sensor Measurement  17','Sensor Measurement  18','Sensor Measurement  19','Sensor Measurement  20',
                    'Sensor Measurement  21']
    
    x_train.columns = column_names
    x_test.columns = column_names
    
    columns_to_drop = ['Unit Number','Cycles', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3',]
#                         'Sensor Measurement  1','Sensor Measurement  5','Sensor Measurement  6','Sensor Measurement  10',
#                         'Sensor Measurement  16','Sensor Measurement  18','Sensor Measurement  19']
    op_setting_columns = ['Cycles', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3']
    
    x_train_data = []
    x_train_op_settings = []
    for i in range(n_flights):
        sample = x_train[x_train['Unit Number']==i+1] 
        sample_op_settings = sample[op_setting_columns]
        sample = sample.drop(columns_to_drop, axis=1)
        
        x_train_data.append(np.asarray(sample))
        x_train_op_settings.append(np.asarray(sample_op_settings))
    x_train = np.asarray(x_train_data)
    x_train_op_settings = np.asarray(x_train_op_settings)
    
    x_test_data = []
    x_test_op_settings = []
    for i in range(n_flights):
        sample = x_test[x_test['Unit Number']==i+1] 
        sample_op_settings = sample[op_setting_columns]
        sample = sample.drop(columns_to_drop, axis=1)
        
        x_test_data.append(np.asarray(sample))
        x_test_op_settings.append(np.asarray(sample_op_settings))
    x_test = np.asarray(x_test_data)
    x_test_op_settings = np.asarray(x_test_op_settings)


    #Checks and deletes sensor data without variation
    sensors_to_delete = []
    for i in range(np.concatenate(x_train).shape[1]):
        if np.max(np.concatenate(x_train)[:,i]) == np.min(np.concatenate(x_train)[:,i]):
            sensors_to_delete.append(i)

    if len(sensors_to_delete) > 0:
        print('Sensors to delete: ', sensors_to_delete)
        for i in range(len(x_train)):
            x_train[i] = np.delete(x_train[i], sensors_to_delete , axis=1)

        for i in range(len(x_test)):
            x_test[i] = np.delete(x_test[i], sensors_to_delete , axis=1)            
                
                
    x_train, y_train = rul(x_train, 0, R_early=R_early)
    x_test, y_test = rul(x_test, y_test, R_early=R_early)

    dataset = {'x_train': x_train,
               'y_train': y_train,
               'x_test': x_test, 
               'y_test': y_test,
               'x_train_op_settings': x_train_op_settings,
               'x_test_op_settings': x_test_op_settings}
    return dataset


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
data_dict = {}

for dataset in datasets: 
    print(dataset)    
    data_dict[dataset] = load_from_txt(dataset)

np.savez('CMAPSS_raw.npz', **data_dict)