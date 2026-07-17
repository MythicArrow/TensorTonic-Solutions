import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Computes a Residual Network block with a projected shortcut connection.
    
    Returns:
    np.ndarray with sum of main path output and projected shortcut after final activation.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Ws = np.asarray(Ws)
    
    def relu(arr):
        return np.maximum(arr, 0)
    
    # 1. Compute the projected shortcut path
    shortcut = x @ Ws
    
    # 2. Compute the main path layer 1
    out = relu(x @ W1)
    
    # 3. Compute the main path layer 2 (without final ReLU yet)
    out = out @ W2
    
    # 4. Add shortcut and apply the final ReLU activation
    y = relu(out + shortcut)
    
    return y