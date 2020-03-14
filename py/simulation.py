import numpy as np
from SpringRank import SpringRank
from scipy.special import gammaln
from py import features

# So I think the general spec we are looking for is that the user can supply a function which gives an arbitrary *matrix* of probabilities, which can then be used for both forward simulation and backward inference. In the case of backward inference, we might also want to ask the user to supply gradients, but we'll get there.

# in both the SIMULATION and the DATA ANALYSIS, agent i is a uniformly random endorser of agent j. 

def increment(A, feature = features.uniform_feature, beta = 0, m_updates = 1, method = 'stochastic', **kwargs):
	'''
	A0 square
	lam in R
	update_fun returns a matrix whose rows are normalized probabilities, of the same shape as A0. 
	m_updates: number of updates to perform in this round
	**kwargs: additional arguments passed to update_fun

	'''

	n = A.shape[0]

	# expected update
	feature_vals = feature(A, **kwargs) # expected update
	bar_delta = features.softmax(feature_vals, beta) 

	# perform stochastic update
	if method == 'stochastic':
		Delta = np.zeros_like(bar_delta) # initialize
		for k in range(m_updates):
			# endorser is uniformly random
			i = np.random.randint(n) 
			# endorsed according to bar_delta[i]
			j = np.random.choice(n, p = bar_delta[i]) 
			Delta[i,j] += 1

	# otherwise, deterministic update: scalar multiple of expected update
	elif method == 'deterministic':
		Delta = bar_delta * m_updates / n

	return Delta

def simulate(A0, n_rounds, lam, feature = features.uniform_feature, beta = 0, m_updates = 1, method = 'stochastic', **kwargs):
	'''
	'''
	n = A0.shape[0]
	
	T = np.zeros((n_rounds, n, n)) # true counts
	T[0] = A0
	A = T.copy() # inferred sequence of state matrices

	# perform sequential udpates
	for k in range(1, n_rounds):
		T[k] = T[k-1]
		Delta = increment(A[k-1], feature = feature, beta = beta, m_updates = m_updates, method = method, **kwargs)
		T[k] += Delta
		A[k] = lam*A[k-1] + (1-lam)*Delta	

	return(T)	







