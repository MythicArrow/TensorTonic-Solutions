import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    def relu(x):
        return x * (x>0)
    h = relu(x @ np.transpose(W1))
    y = relu(h @ np.transpose(W2) + x)
    return y
