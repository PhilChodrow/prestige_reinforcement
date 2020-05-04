import numpy as np

def matrix_sort(A, v):
    row_sorted = A[np.argsort(v)]
    col_sorted = row_sorted[:, np.argsort(v)]
    return(col_sorted)	