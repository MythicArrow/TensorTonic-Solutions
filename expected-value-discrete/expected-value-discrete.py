import numpy as np

def expected_value_discrete(x, p):
    """
    Returns the expected value of a discrete random variable.
    
    Parameters:
    x : array-like, outcomes
    p : array-like, probabilities corresponding to each outcome
    
    Returns:
    float: expected value
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    p = np.asarray(p)
    
    # Validation: probabilities must sum to 1
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.0")
    
    # Validation: arrays must have the same shape
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")
    
    # Calculate expected value: E[X] = sum(x * p)
    prod = np.multiply(x, p)
    exp_val = np.sum(prod)
    
    return float(exp_val)