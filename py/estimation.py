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
	# print(ll)
	return(ll)

def ML_params(T, S, b0, bounds = None):

	res = minimize(
		fun = lambda b: -ll(T, S, b),
		x0 = b0,
		bounds = bounds
		)
	return(res)

def ML(T, A0, feature_fun, dim = 1, **kwargs):
	'''
	**kwargs are passed to the optimization over lambda. 
	It's not necessary to control the optimization over the params 
	because the objective is convex. 
	'''
	
	def obj(lam):
		A = state_matrix(T, A0 = A0, lam = lam)
		S = feature_fun(A)	
		res = ML_params(T, S, b0 = np.zeros(dim))
		out = res['fun']
		b0 = res['x']
		# print(lam, out)
		return(out)
	print('computing memory hyperparameter lambda')
	RES = minimize(fun = obj, 
					x0 = np.array([0.0]),  
					**kwargs)

	lam = RES['x']
	hess_inv = RES['hess_inv']

	# if type(hess_inv) == 'LbfgsInvHessProduct':
		# hess_inv = hess_inv.todense()

	# lam_stderr = np.sqrt(hess_inv) 

	print('computing parameter vector beta')
	A = state_matrix(T, A0 = A0, lam = lam)
	S = feature_fun(A)
	res = ML_params(T, S, b0 = np.zeros(dim))

	beta = res['x']
	beta_stderr = np.sqrt(np.diag(res['hess_inv']))

	return({
		'lam' : lam,
		# 'lam_stderr' : lam_stderr,
		'beta' : beta,
		'beta_stderr' : beta_stderr,
		'LL' : - res['fun']
		}) 

def estimate_hessian(T, A0, feature_fun, lam, beta):

	def f(par_vec):
		A = state_matrix(T, A0 = A0, lam = par_vec[0])
		S = feature_fun(A)	
		return(ll(T, S, np.array([par_vec[1], par_vec[2]])))

	return Hessian(f)(np.array([lam[0], beta[0], beta[1]]))
		



