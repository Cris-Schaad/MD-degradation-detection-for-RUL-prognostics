import numpy as np  

def time_window_sampling(x, y, time_window, 
                         temporal_axis=0, 
                         time_step = 1, 
                         y_time_lag = 0, 
                         add_last_dim = False, 
                         time_step_flatten = False):
        
    framed_x = []
    framed_y = []
    for x_sample, y_sample in zip(x, y):
        
        indx = x_sample.shape[temporal_axis] - y_time_lag
        indx_left = True if indx - time_window >= 0 else False
        
        if indx_left == True:
            framed_x_sample = []
            framed_y_sample = []
            
            x_sample = np.swapaxes(x_sample, 0, temporal_axis)
            while indx_left:
                
                time_step_data = x_sample[indx-time_window:indx] 
                time_step_data = np.swapaxes(time_step_data, temporal_axis, 0)
                if time_step_flatten:
                    time_step_data = time_step_data.flatten()
                    
                framed_x_sample.append(time_step_data)
                framed_y_sample.append(y_sample[indx+y_time_lag-1])
                indx = indx-time_step

                if indx-time_window <= 0:
                    indx_left=False
            
            if add_last_dim: 
                framed_x_sample = np.expand_dims(np.asarray(framed_x_sample), axis=-1)
            else:
                framed_x_sample = np.asarray(framed_x_sample)
            
            if time_window ==1:
                framed_x_sample = np.squeeze(framed_x_sample, axis=1)
            
            framed_x.append(framed_x_sample)
            framed_y.append(np.asarray(framed_y_sample))
    return np.asarray(framed_x), np.asarray(framed_y)



def op_time_add(x, y_original, return_RUL=False, expand_y_dims=False, mode='array_of_samples'):

    y = np.copy(y_original)
    if mode == 'array_of_samples':
        for i, sample in enumerate(x):  
            len_sample = len(sample)
            op_time_sample = np.asarray([j+1 for j in range(len_sample)])
            x[i] = np.column_stack((sample, op_time_sample))
            if return_RUL:
                y[i] = np.expand_dims(op_time_sample[::-1], axis=-1)
            if expand_y_dims:
                y[i] = np.expand_dims(y[i], axis=-1)
    
    if mode == 'sample':
        op_time_sample = np.asarray([j+1 for j in range(len(x))])
        x = np.column_stack((x, op_time_sample))
        if return_RUL:
            y = np.expand_dims(op_time_sample[::-1], axis=-1)
        if expand_y_dims:
            y = np.expand_dims(y, axis=-1)
    return x, y



def samples_reverse(x):
    
    for i in range(len(x)):
        x[i] = x[i][::-1]
    return x



def rul(x, max_rul=np.inf, normalized=False, normalizing_factor=1):
       
    y = []
    for i in range(len(x)):
        sample = x[i]        
        rul_sample = np.asarray([i for i in range(len(sample))])
        if normalized:
            rul_sample = rul_sample/np.max(rul_sample)
        rul_sample = rul_sample[::-1]/normalizing_factor
        
        rul_sample = np.where(rul_sample<max_rul, rul_sample, max_rul)
        y.append(np.expand_dims(rul_sample, axis=-1))   
    return x, np.asarray(y)



def FEMTO_bearing_op_settings(sample, sample_name):
    
    condition_1 = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']
    condition_2 = ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7']
    condition_3 = ['Bearing3_1', 'Bearing3_2', 'Bearing3_3']
  
    if sample_name in condition_1:
        sample_rpm = 1800
        sample_force = 4000

    if sample_name in condition_2:
        sample_rpm = 1650
        sample_force = 4200
    
    if sample_name in condition_3:
        sample_rpm = 1500
        sample_force = 5000
        
    deg_time = np.asarray([10*i for i in  range(len(sample))])
    rpm = np.asarray([sample_rpm for i in range(len(sample))])
    force = np.asarray([sample_force for i in range(len(sample))])    
    return deg_time, rpm, force



def FEMTO_op_settings_add(x_train, y_train, x_test, y_test, train_samples_names, test_sample_names):

    for i in range(len(x_train)):
        deg_time, rpm, force = FEMTO_bearing_op_settings(x_train[i] ,train_samples_names[i])
        x_train[i] = np.column_stack((x_train[i], deg_time, rpm, force))
          
    deg_time, rpm, force = FEMTO_bearing_op_settings(x_test, test_sample_names)
    x_test = np.column_stack((x_test, deg_time, rpm, force))     
    return x_train, y_train, x_test, y_test
