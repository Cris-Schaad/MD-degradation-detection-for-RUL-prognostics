import os
import numpy as np
import matplotlib.pyplot as plt

import MS_iterator as MS_iterator
import data_processing as dp


plt.close('all')


datasets = ['FD001', 'FD002', 'FD003', 'FD004']
data_dir = os.path.join('C-MAPSS', 'processed_data')
dataset_raw_npz = dict(np.load(os.path.join(data_dir,'CMAPSS_raw.npz'), allow_pickle=True))


time_window = 20
initial_healthy_cycles = -1

data_dict = {}
for i, dataset in enumerate(datasets): 
    
    print('Processing dataset: ', dataset)
    dataset_dict = dataset_raw_npz[dataset][()]
    x_train = dataset_dict['x_train']
    y_train = dataset_dict['y_train']
    x_test = dataset_dict['x_test']
    y_test = dataset_dict['y_test']
    
    #Degradation detector
    k = 2; n = 6; sigma = 1
    start_period = 10
    iterations = 10


    iterator = MS_iterator.iterator(k, n, sigma, initial_healthy_cycles, start_period)
    m_d, detector, threshold, deg_start_ind = iterator.iterative_calculation(x_train, n_iterations=iterations, 
                                                                             verbose=False)

    plt.figure()
    plt.plot(np.asarray(iterator.iter_ms_dim)/len(np.concatenate(x_train)))


    x_train_md = np.asarray([m_d.fit(i)/np.mean(m_d.fit(i)[:start_period]) for i in x_train])
    x_test_md = np.asarray([m_d.fit(i)/np.mean(m_d.fit(i)[:start_period]) for i in x_test])
    
    plt.figure()
    for i in x_train_md:
        plt.plot(i)
        
    # Training set   
    ruls = []
    for i, sample in enumerate(x_train_md):
        ruls.append(len(sample) - deg_start_ind[i] +1)

    print('Average RUL: {:.2f}'.format(np.mean(ruls)))
    print('Min RUL: {:.2f}'.format(np.min(ruls)))
    print('Max RUL: {:.2f} \n'.format(np.max(ruls)))
    

    # Test set
    test_deg_start_ind = []
    for i, sample in enumerate(x_test_md):
        indx = detector.detect(sample)
        test_deg_start_ind.append(indx)
            
    test_ruls = []; ignored_test_ruls = []
    for i, sample_y in enumerate(y_test):
        if test_deg_start_ind[i] < len(sample_y)-1:
            test_ruls.append(np.min(sample_y))
        else:
            ignored_test_ruls.append(np.min(sample_y))
    print('Total test samples left: {}\n'.format(len(test_ruls)))
        
    plt.figure()
    plt.hist(test_ruls, 20, range=(0,200), color='r', alpha=0.5)
    plt.hist(ignored_test_ruls, 20, range=(0,200), color='g', alpha=0.5)
    
    
    for i in range(len(x_train)):
        x_train[i] = np.column_stack((x_train[i], x_train_md[i]))
    for i in range(len(x_test)):
        x_test[i] = np.column_stack((x_test[i], x_test_md[i]))
    
        
    # Sampling from degradation start index
    x_train, y_train = detector.sampling_from_index(x_train, y_train, deg_start_ind, time_window)
    x_test, y_test = detector.sampling_from_index(x_test, y_test, test_deg_start_ind, time_window)
 
    
    #Time window sampling 
    x_train, y_train = dp.time_window_sampling(x_train, y_train, time_window, add_last_dim=False)
    x_test, y_test = dp.time_window_sampling(x_test, y_test, time_window, add_last_dim=False)
    

    data_dict[dataset] =  {'x_train': x_train,
                          'y_train':y_train,
                          'x_test': x_test, 
                          'y_test': y_test}

np.savez(os.path.join(data_dir,'CMAPSS_dataset.npz'),
          dataset = data_dict)