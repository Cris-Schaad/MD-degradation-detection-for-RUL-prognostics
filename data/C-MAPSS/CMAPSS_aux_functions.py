import numpy as np

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