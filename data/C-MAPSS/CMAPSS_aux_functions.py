import numpy as np
import matplotlib.pyplot as plt


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


def correlation(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    x_sum = np.sum((x-mean_x)**2)
    y_sum = np.sum((y-mean_y)**2)
    return np.dot(x-mean_x, y-mean_y)/np.sqrt(x_sum*y_sum)


def correlation_matrix(x):
    
    n_vars = x.shape[1]    
    corr = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):                 
            corr[i,j] = correlation(x[:,i], x[:,j])
    return corr


def corr_matrix_plot(x):
    cor = correlation_matrix(x)
    mask = np.triu(np.ones_like(cor))
    cor = np.ma.masked_where(mask, cor)
    
    sensors_to_delete = []
    for i in range(1, cor.shape[0]):
        var_max_cor = np.max(np.abs(cor[i]))
        print('Variable max corr: {:.4f}'.format(var_max_cor))
        if var_max_cor >= 0.9999:
            sensors_to_delete.append(i)
    
    # cor_no_diagonal = []
    # for i in range(len(cor)):
    #     cor_no_diagonal.append(np.delete(cor[i].flatten(), obj=[i]))
    # cor = np.column_stack(cor_no_diagonal)
        
    
    plt.figure()
    plt.imshow(cor)
    plt.colorbar()
    
    return sensors_to_delete