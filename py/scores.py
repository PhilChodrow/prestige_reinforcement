import numpy as np

from numba import jit
from SpringRank import SpringRank

SpringRank_score = SpringRank

@jit(nopython=True)
def homebrew_SpringRank_score(A, alpha = 10**(-8)):
    Di = np.diag(A.sum(axis = 0))
    Do = np.diag(A.sum(axis = 1))
    L = Di + Do - A - A.T + alpha*np.eye(A.shape[0])
    return(np.linalg.inv(L).dot(Do - Di).dot(np.ones(A.shape[0])))

@jit(nopython=True)
def powered_degree_score(A,p):
	return(A.sum(axis = 0)**p)

@jit(nopython=True)
def fiedler_vector_score(A):
    
    # form undirected Laplacian
    A = (A + A.T)/2
    D = np.diag(A.sum(axis = 1))
    L = D - A

    eig = np.linalg.eigh(L)
    v = eig[1][:,1]    
    v = v / np.sqrt((v**2).sum())

    return(v)	

@jit(nopython=True)
def katz_score(A, alpha = .001):
    n = A.shape[0]
    e = np.ones(n)
    I = np.eye(n,n)
    return(np.linalg.inv(I - alpha*A).dot(e))

@jit(nopython=True)
def PageRank_score(A, n_iter = 30, alpha = 0.15):
    n = A.shape[0]
    e = np.ones(n)
    d = A.sum(axis = 1)
    norm = np.outer(e, d)
    A_ = A/norm
    
    M = (1-alpha)*A_ + alpha/n
    
    v = np.random.rand(n)
    for i in range(n_iter):
        v = M.dot(v)
    if v[0] < 0:
        v = -v
    return(v/v.sum())
