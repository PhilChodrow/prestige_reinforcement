import numpy as np

from numba import jit
from SpringRank import SpringRank

SpringRank_score = SpringRank
'''
Simple alias for SpringRank as available on pip. In both versions of SpringRank, it is usually desirable to compute the score on A.T rather than on A. 
'''

@jit(nopython=True)
def homebrew_SpringRank_score(A, alpha = 10**(-8)):
    '''
    When it is possible to assume that alpha > 0, we can give a faster version of the SpringRank score with vanilla numpy and numba. 
    In both versions of SpringRank, it is usually desirable to compute the score on A.T rather than on A. 
    '''
    Di = np.diag(A.sum(axis = 0))
    Do = np.diag(A.sum(axis = 1))
        
    L = Di + Do - A - A.T + alpha*np.eye(A.shape[0])
    return(np.linalg.inv(L).dot(Do - Di).dot(np.ones(A.shape[0])))

@jit(nopython=True)
def powered_degree_score(A,p = 1):
    '''
    Computes the column sum of A, optionally raised to a user-specified power. 
    The power can also be set as a feature map rather than as an intrinsic part of the score function. 
    '''
    return(A.sum(axis = 0)**p)

@jit(nopython=True)
def fiedler_vector_score(A):
    '''
    Compute the Fiedler unit vector of A: the Perron-Frobenius eigenvector with largest (real) eigenvalue corresponding to the symmetrized, unnormalized Laplacian of A. 
    '''
    
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
    '''
    Compute the Katz centrality vector of A with given alpha. In our context, we usually want to compute on A.T 
    '''
    n = A.shape[0]
    e = np.ones(n)
    I = np.eye(n,n)
    return(np.linalg.inv(I - alpha*A).dot(e))

@jit(nopython=True)
def PageRank_score(A, n_iter = 30, alpha = 0.85):
    '''
    Approximate the PageRank score of A with specified teleportation parameter alpha via the power method with specified number of iterations. 
    '''
    n = A.shape[0]
    e = np.ones(n)
    d = A.sum(axis = 1)
    D_inv = np.diag(1/d)
    # norm = np.outer(e, d)
    P = np.dot(A.T, D_inv)
    
    M = (alpha)*P + (1-alpha)/n
    
    v = np.random.rand(n)
    for i in range(n_iter):
        v = M.dot(v)
    if v[0] < 0:
        v = -v
    return(v/v.sum())

@jit(nopython=True)
def eigenvector_score(A, n_iter = 30):
    '''
    Compute the eigenvector centrality of A. In our examples, we usually want to compute the score on A.T. 
    '''
    n = A.shape[0]
    v = np.random.rand(n)
    for i in range(n_iter):
        v = A.dot(v)
        v = v / v.sum()
    return(v)

# @jit(nopython=True)
def RW_score(A, p = .75, alpha = 0):
    '''
    Compute the Random Walker Ranking score of Callaghan, Porter, and Mucha, described here: 

    https://arxiv.org/abs/physics/0310148
    '''
    wins = A.T
    losses = A
    D_wins = np.diag(wins.sum(axis = 1))
    D_losses = np.diag(losses.sum(axis = 1))
    M = p*(wins - D_losses) + (1-p)*(losses - D_wins) + alpha*np.eye(A.shape[0])
    eigs = np.linalg.eig(M)
    which_eig = np.argmin(np.abs(eigs[0]))
    val = eigs[0][which_eig]
    v = eigs[1][:,which_eig]
    v = np.abs(v)
    return(v)