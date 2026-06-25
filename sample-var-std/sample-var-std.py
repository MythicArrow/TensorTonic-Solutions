import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    n = len(x)
    x = np.stack(x)
    mean = np.mean(x)
    var= np.sum((x-mean)**2) / (n-1)
    dev = np.sqrt(var)
    return var, dev
    
    