import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    
    Parameters:
    x : array-like
        Sample data
    mu0 : float
        Hypothesized population mean
    
    Returns:
    float
        t-statistic
    """
    x = np.asarray(x)
    n = len(x)
    
    if n < 2:
        raise ValueError("Sample size must be at least 2")
    
    mean = np.mean(x)
    # Calculate sample standard deviation
    variance = np.sum((x - mean)**2) / (n - 1)
    s = np.sqrt(variance)
    
    # Calculate t-statistic
    t = (mean - mu0) / (s / np.sqrt(n))
    
    return float(t)