import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 1:
        max_x = np.max(x)
        exp_x = np.exp(x- max_x)
        return exp_x / np.sum(exp_x)
    elif x.ndim == 2:
        max_x2 = np.max(x, axis=1, keepdims= True)
        exp_x2 = np.exp(x-max_x2)
        return exp_x2 / np.sum(exp_x2, axis=1, keepdims = True)