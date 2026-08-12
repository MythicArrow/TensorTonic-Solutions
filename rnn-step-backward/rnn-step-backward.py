import numpy as np

def rnn_step_backward(dh, cache):
    # Unpack the cache
    x_t, h_prev, h_t, W, U, b = cache
    
    # Ensure arrays
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    b = np.asarray(b)
    
    # 1. Derivative of tanh
    # dh: (H,), h_t: (H,) -> dtanh: (H,)
    dtanh = dh * (1 - h_t**2)
    
    # 2. Gradients for weights
    # dW: (H, D) = outer(dtanh (H,), x_t (D,))
    dW = np.outer(dtanh, x_t)
    
    # dU: (H, H) = outer(dtanh (H,), h_prev (H,))
    dU = np.outer(dtanh, h_prev)
    
    # db: (H,) = sum(dtanh)
    db = dtanh # If single sample, sum is just the vector itself if axis logic is handled elsewhere, 
               # but strictly for a single step, db = dtanh is correct if b is added element-wise.
               # If batched, you'd sum over axis 0. Assuming single step here.
    
    # 3. Gradients for inputs
    # dx_t: (D,) = W.T (D, H) @ dtanh (H,)
    # FIX: Order matters. W.T is (D, H), dtanh is (H,). Result is (D,)
    dx_t = W.T @ dtanh
    
    # dh_prev: (H,) = U.T (H, H) @ dtanh (H,)
    dh_prev = U.T @ dtanh
    
    return dx_t, dh_prev, dW, dU, db