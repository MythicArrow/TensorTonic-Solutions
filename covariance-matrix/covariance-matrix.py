import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    shape = X.shape
    n_samples = shape[0]
    dim = X.ndim
    if n_samples<2 or dim!=2:
        return None
    first = np.mean(X, axis=0)
    centered = X - first
    norm_factor = 1 / (n_samples-1)
    return norm_factor * np.matmul(np.transpose(centered),centered)