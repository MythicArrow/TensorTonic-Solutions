import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Convert input to a NumPy array while keeping its original shape
    x = np.array(x, dtype=float)
    
    # Vectorize math.erf so it accepts NumPy arrays
    vec_erf = np.vectorize(math.erf)
    
    # Compute the exact GELU formula
    return 0.5 * x * (1.0 + vec_erf(x / math.sqrt(2.0)))

