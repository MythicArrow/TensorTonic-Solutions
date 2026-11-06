import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    dot_prod = np.dot(a,b)
    a_euclid = np.linalg.norm(a)
    b_euclid = np.linalg.norm(b)
    if a_euclid == 0.0:
        return 0.0
    if b_euclid == 0.0:
        return 0.0
    cosine = dot_prod / (a_euclid * b_euclid)
    return float(cosine)