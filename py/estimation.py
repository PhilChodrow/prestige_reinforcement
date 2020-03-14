import numpy as np
from scipy.special import gammaln # gammaln(x+1) = log x!
from SpringRank import SpringRank
from py import features 

def state_matrix(T, lam, A0 = None):
    n_rounds = T.shape[0]
    DT = np.diff(T, axis = 0)
    A = np.zeros_like(T)
    if A0 is None:
        A[0] = T[0]
    else:
        A[0] = A0
    for j in range(1,n_rounds):
        A[j] = lam*A[j-1]+(1-lam)*DT[j-1]
    return(A)

def ll(T, S, beta):
	'''
	so what we'd like to do here is separate graph computations from parameter optimization in a principled way. The point is that, having computed the features, the optimization over beta is convex and hopefully fast. 

	features is a hyperarray with axes time x n_features x i x j
	beta is a vector. 

	'''
	n_rounds, n = T.shape[0], T.shape[1]
	DT = np.diff(T, axis = 0)

	gamma = features.softmax(S, beta) 

	ll = (DT*np.log(gamma[:-1])).sum()
	return(ll)

