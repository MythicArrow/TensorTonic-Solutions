import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Convert input to a NumPy array if it isn't one already
    x = np.asarray(x)
    
    # Target the last two dimensions (Height and Width) dynamically
    if x.ndim not in (3,4):
        raise ValueError
    return np.mean(x, axis=(-2, -1))
