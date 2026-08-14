import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    state = {
        'min': np.full(D, np.inf),
        'max': np.full(D, -np.inf)
    }
    return state

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, and return the normalized batch.
    Assumes X_batch has shape (N, D), where N is batch size and D is features.
    """
    # 1. Compute feature-wise min and max for the current batch
    batch_min = np.amin(X_batch, axis=0)
    batch_max = np.amax(X_batch, axis=0)
    
    # 2. Update global running tracking states
    state['min'] = np.minimum(state['min'], batch_min)
    state['max'] = np.maximum(state['max'], batch_max)
    
    # 3. Scale current batch dynamically using updated global states
    normalized_x = (X_batch - state['min']) / (state['max'] - state['min'] + eps)
    
    return normalized_x