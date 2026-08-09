import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    rows, cols = A.shape
    transposed = np.zeros((cols, rows), dtype=A.dtype)
    for r in range(rows):
        for c in range(cols):
            transposed[c, r] = A[r, c]
    return transposed
   
