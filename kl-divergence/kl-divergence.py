import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.stack(p)
    q = np.stack(q)
    q_stable = q + eps
    return np.sum(p*np.log(p/q_stable))
    