import numpy as np
from scipy.stats import skew, kurtosis


def features(x):
    rms = np.sqrt(np.mean(x**2, axis=0))
    ptp = np.ptp(x, axis=0)
    cf = abs(ptp)/rms
    mean = np.mean(x, axis=0)
    maxi = np.max(abs(x), axis=0)
    var = np.var(x, axis=0)
    skw = skew(x, axis=0)
    kurt = kurtosis(x, axis=0)

    output_features = np.column_stack((rms,ptp,cf,mean,maxi,var,skw,kurt))
    return output_features.flatten()


def super_features(x):
    return np.hstack((features(x), features(np.diff(x)), features(x[1:]+x[:-1])))


def windows(x, time_window, time_step):
    
    wins = []
    indx = len(x)
    indx_left = True
    while indx_left:
        sample = x[indx-time_window:indx]
        wins.append(sample[::-1])
        indx = indx-time_step
        if indx-time_window < 0:
            indx_left=False
    return np.asarray(wins[::-1])