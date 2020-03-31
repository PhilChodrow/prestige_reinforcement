import numpy as np
from scipy.special import gammaln # gammaln(x+1) = log x!
from SpringRank import SpringRank
from py import features 
from scipy.optimize import minimize
from py.features import SR_quadratic_feature
from numdifftools import Hessian


def state_matrix(T, lam, A0 = None):
    n_rounds = T.shape[0]
    DT = np.diff(T, axis = 0)
    A = np.zeros_like(T).astype(float)
    if A0 is None:
        A[0] = T[0]
    else:
        A[0] = A0
    for j in range(1,n_rounds):
        A[j] = lam*A[j-1]+(1-lam)*DT[j-1]
    return(A)


def estimate_hessian(T, A0, feature_fun, lam, beta):

	def f(par_vec):
		A = state_matrix(T, A0 = A0, lam = par_vec[0])
		S = feature_fun(A)	
		return(ll(T, S, np.array([par_vec[1], par_vec[2]])))

	return Hessian(f)(np.array([lam[0], beta[0], beta[1]]))
		



