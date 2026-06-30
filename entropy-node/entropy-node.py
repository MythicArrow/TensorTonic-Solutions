import numpy as np

def entropy_node(y):
    """
    Compute Shannon entropy for a single node using stable logarithms.
    
    Parameters:
    y (array-like): The target values or classes at the node.
    
    Returns:
    float: The entropy of the node.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
        
    # Get unique classes and their counts
    ohio, counts = np.unique(y, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(y)
    
    # Filter out 0 probabilities to avoid log2(0) = -inf errors
    probabilities = probabilities[probabilities > 0]
    
    # Compute the entropy: -Sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy
