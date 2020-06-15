import os
import numpy as np
import pandas as pd
from CMAPSS_utils import close_all
from CMAPSS_utils import get_data
from CMAPSS_utils import rul


close_all()
data_dict = {}

datasets = ['FD001', 'FD002', 'FD003', 'FD004']
for dataset in datasets: 
    print(dataset)    

    path = os.getcwd()
    x_train = pd.read_csv(os.path.join(path,'raw_data','train_'+dataset+'.txt'), delim_whitespace=True, header=None)
    x_test = pd.read_csv(os.path.join(path,'raw_data','test_'+dataset+'.txt'),delim_whitespace=True,header=None)    
    y_test = np.array(pd.read_csv(os.path.join(path,'raw_data','RUL_'+dataset+'.txt'), delim_whitespace=True, header=None))
    
    op_names = ['Unit Number','Cycles', 'Operational Setting 1', 'Operational Setting 2', 'Operational Setting 3']
    sensor_names = ['Sensor Measurement  1', 'Sensor Measurement  2','Sensor Measurement  3','Sensor Measurement  4',
                    'Sensor Measurement  5','Sensor Measurement  6','Sensor Measurement  7','Sensor Measurement  8',
                    'Sensor Measurement  9','Sensor Measurement  10','Sensor Measurement  11','Sensor Measurement  12',
                    'Sensor Measurement  13','Sensor Measurement  14','Sensor Measurement  15','Sensor Measurement  16',
                    'Sensor Measurement  17','Sensor Measurement  18','Sensor Measurement  19','Sensor Measurement  20',
                    'Sensor Measurement  21']
    
    x_train.columns = op_names+sensor_names
    x_test.columns = op_names+sensor_names
    
    # Data drop
    x_train, x_train_op_settings = get_data(x_train, op_names[1:], op_names)
    x_test, x_test_op_settings = get_data(x_test, op_names[1:], op_names)


    # RUL labelling.
    x_train, y_train = rul(x_train, 0, R_early = np.inf)
    x_test, y_test = rul(x_test, y_test, R_early = np.inf)
    
    
    #Checks and deletes sensor data without variation
    sensors_to_delete = []
    for i in range(np.concatenate(x_train).shape[1]):
        if len(np.unique(np.concatenate(x_train)[:,i])) <= 5:
            sensors_to_delete.append(i)
            
    if dataset == 'FD003':
        sensors_to_delete = [0, 4, 5, 9, 15, 17, 18]
    
    # sensors_to_delete = [0, 4, 5, 9, 15, 17, 18]
    if len(sensors_to_delete) > 0:
        print('Sensors to delete: ', [i+1 for i in sensors_to_delete])
        x_train = np.asarray([np.delete(i, sensors_to_delete , axis=1) for i in x_train])
        x_test = np.asarray([np.delete(i, sensors_to_delete , axis=1) for i in x_test])     
    
    
    
    data_dict[dataset] = {'x_train': x_train,
                          'y_train': y_train,
                          'x_test': x_test, 
                          'y_test': y_test,
                          'x_train_op_settings': x_train_op_settings,
                          'x_test_op_settings': x_test_op_settings}

np.savez(os.path.join('processed_data','CMAPSS_raw.npz'), **data_dict)