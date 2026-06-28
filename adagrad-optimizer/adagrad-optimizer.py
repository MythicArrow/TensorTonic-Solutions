import numpy as np

def adagrad_step(w, dw, cache, lr=0.01, eps=1e-8):
    """
    w: weights (numpy array)
    dw: gradients (numpy array)
    cache: historical accumulated gradients (numpy array)
    lr: learning rate
    eps: smoothing term to avoid division by zero
    """
    w = np.asarray(w, dtype=np.float64)
    dw = np.asarray(dw, dtype=np.float64)
    cache = np.asarray(cache, dtype=np.float64)
    # 1. Standard gradient square accumulation
    cache += dw ** 2
    
    # 2. Enforce the gt-1 constraint on the cache
    # This ensures that the effective historical gradient modifier is >= 1
    #effective_cache = np.maximum(cache, 1.0)
    
    # 3. Parameter update
    w -= lr / np.sqrt(cache+eps) * dw
    
    return w, cache
