import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    
    Parameters:
    n (int): Number of trials
    p (float): Success probability (0 <= p <= 1)
    k (int): Number of successes (0 <= k <= n)
    
    Returns:
    tuple: (pmf, cdf) as floats
    """
    # Handle edge cases for k
    if k < 0:
        return 0.0, 0.0
    if k >= n:
        # If k >= n, CDF is 1.0 (probability of <= n successes is 1)
        # PMF needs to be calculated for k=n specifically
        # Note: If k > n, PMF is 0.0, but problem constraints say 0 <= k <= n
        if k > n:
            return 0.0, 1.0
        
        # Calculate PMF for k=n
        # comb(n, n) is 1, p^n * (1-p)^0
        pmf = comb(n, k, exact=False) * (p ** k) * ((1 - p) ** (n - k))
        return float(pmf), 1.0

    # Calculate PMF for the specific k
    # Use exact=False for float output, which is faster for large n
    pmf = comb(n, k, exact=False) * (p ** k) * ((1 - p) ** (n - k))
    
    # Calculate CDF: Sum of PMFs from i=0 to k
    cdf = 0.0
    
    # Loop from 0 to k (inclusive)
    for i in range(k + 1):
        # Calculate PMF for each i and add to cdf
        term = comb(n, i, exact=False) * (p ** i) * ((1 - p) ** (n - i))
        cdf += term
        
    return float(pmf), float(cdf)