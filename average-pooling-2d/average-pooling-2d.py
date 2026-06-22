import numpy as np
def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    X = np.stack(X)
    height, width = X.shape
    out_h = height // pool_size
    out_w = width // pool_size
    out_arr = np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            block_sum = 0.0

            for a in range(pool_size):
                for b in range(pool_size):
                    block_sum += X[i * pool_size + a][j * pool_size + b]

            out_arr[i, j] = block_sum / (pool_size*pool_size)
    return out_arr.tolist()
    