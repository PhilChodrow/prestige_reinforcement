import numpy as np
from SpringRank import SpringRank


def softmax(S, beta):
	'''
	v a vector
	beta a scalar
	'''
	if len(S.shape) == 3: # S represents a single timestep
		# S = S[np.newaxis, :,:,:] 
		phi = np.tensordot(beta, S, axes = (0,0))
		gamma = np.exp(phi)
		gamma = gamma / gamma.sum(axis = 1)[:,np.newaxis]

		return(gamma)

	elif len(S.shape) == 4: # S is a series of sets of features
		n_rounds = S.shape[0]
		gamma = np.zeros((n_rounds,S.shape[2], S.shape[3]))

		for i in range(0, n_rounds):
			gamma[i] = softmax(S[i], beta)

	return(gamma)

def time_series_vectorizer(feature_fun, **kwargs):
	'''

	'''

	def vectorized_feature(A, **kwargs):
		if len(A.shape) == 2:
			return(feature_fun(A, **kwargs))
		
		elif len(A.shape) == 3: 
			n_rounds = A.shape[0]
			S = np.zeros((n_rounds, 2, A.shape[1], A.shape[1]))

			for i in range(0,n_rounds):
				S[i] = feature_fun(A[i], **kwargs)

		return(S)

	return(vectorized_feature)

# -----------------------------------------------------------------------------
# Specific Update Functions
# -----------------------------------------------------------------------------

def SR_quadratic_feature_(A, alpha = 10**(-3)):

	n = A.shape[1]
	np.seterr(divide='ignore', invalid='ignore') 
	s = SpringRank.SpringRank(A.T, alpha = alpha)

	S = np.zeros((2,n,n))

	S[0] = np.tile(s, (n,1))
	S[1] = (S[0] - S[0].T)**2

	return(S)

SR_quadratic_feature = time_series_vectorizer(SR_quadratic_feature_)

def SR_linear_feature_(A, alpha = 10**(-3)):
	n = A.shape[1]
	np.seterr(divide='ignore', invalid='ignore') 
	s = SpringRank.SpringRank(A.T, alpha = alpha)

	S = np.zeros((2,n,n))

	S[0] = np.tile(s, (n,1))
	S[1] = 1

	return(S)

SR_linear_feature = time_series_vectorizer(SR_linear_feature_)

def degree_linear_feature_(A, d0 = 1):

	n = A.shape[1]
	
	S = np.zeros((2,n,n))

	S[0] = np.sqrt(A.sum(axis = 0) + d0)
	S[1] = 1

	return(S)

degree_linear_feature = time_series_vectorizer(degree_linear_feature_)

def degree_quadratic_feature_(A, d0 = 1):

	n = A.shape[1]
	S = np.zeros((2,n,n))
	S[0] = np.sqrt(A.sum(axis = 0) + d0)
	S[1] = (S[0] - S[0].T)**2

	return(S)

degree_quadratic_feature = time_series_vectorizer(degree_quadratic_feature_)

def uniform_feature_(A, k_features = 1, **kwargs):
	
	S = np.zeros((k_features, A.shape[0], A.shape[1]))
	return(S)

uniform_feature = time_series_vectorizer(uniform_feature_)



# # def SR_quadratic_feature(A, alpha = 10**(-3)):
# # 	'''

# # 	'''
# # 	n = A.shape[1]

# # 	if len(A.shape) == 2: # A is a single timestep 
	
# # 		np.seterr(divide='ignore', invalid='ignore') 
# # 		s = SpringRank.SpringRank(A.T, alpha = alpha)

# # 		S = np.zeros((2,n,n))

# # 		S[0] = np.tile(s, (n,1))
# # 		S[1] = (S[0] - S[0].T)**2

# # 		return(S)

# # 	elif len(A.shape) == 3: # A is a series of timesteps
# # 		n_rounds = A.shape[0]
# # 		S = np.zeros((n_rounds, 2, n, n))

# # 		for i in range(0,n_rounds):
# # 			S[i] = SR_quadratic_feature(A[i], alpha)

# # 	return(S)

# def SR_linear_feature(A, alpha = 10**(-3)):
# 	'''
# 	'''
# 	n = A.shape[1]

# 	if len(A.shape) == 2: # A is a single timestep 
	
# 		np.seterr(divide='ignore', invalid='ignore') 
# 		s = SpringRank.SpringRank(A.T, alpha = alpha)

# 		S = np.zeros((2,n,n))

# 		S[0] = np.tile(s, (n,1))
# 		S[1] = 1

# 		return(S)

# 	elif len(A.shape) == 3: # A is a series of timesteps
# 		n_rounds = A.shape[0]
# 		S = np.zeros((n_rounds, 2, n, n))

# 		for i in range(0,n_rounds):
# 			S[i] = SR_linear_feature(A[i], alpha)

# 	return(S)

# def degree_linear_feature(A, d0 = 1):
# 	'''
# 	'''
# 	n = A.shape[1]

# 	if len(A.shape) == 2: # A is a single timestep 
	
# 		S = np.zeros((2,n,n))

# 		S[0] = np.sqrt(A.sum(axis = 0) + d0)
# 		S[1] = 1

# 		return(S)

# 	elif len(A.shape) == 3: # A is a series of timesteps
# 		n_rounds = A.shape[0]
# 		S = np.zeros((n_rounds, 2, n, n))

# 		for i in range(0,n_rounds):
# 			S[i] = degree_linear_feature(A[i], d0 = d0)

# 	return(S)

# def degree_quadratic_feature(A, d0 = 1):
# 	'''

# 	'''
# 	n = A.shape[1]
# 	if len(A.shape) == 2: # A is a single timestep 
	
# 		S = np.zeros((2,n,n))

# 		S[0] = np.sqrt(A.sum(axis = 0) + d0)
# 		S[1] = (S[0] - S[0].T)**2

# 		return(S)

# 	elif len(A.shape) == 3: # A is a series of timesteps
# 		n_rounds = A.shape[0]
# 		S = np.zeros((n_rounds, 2, n, n))

# 		for i in range(0,n_rounds):
# 			S[i] = degree_quadratic_feature(A[i], d0)

# 	return(S)






# # def SR_linear_quadratic_dynamics(A, beta, eta, alpha = 0):
# # 	'''
# # 	returns a matrix
# # 	'''
# # 	np.seterr(divide='ignore', invalid='ignore') 
# # 	s = SpringRank.SpringRank(A.T, alpha = alpha)

# # 	e = np.ones_like(s)
# # 	S = np.outer(s, e)
# # 	phi = beta*S.T + eta*(S - S.T)**2

# # 	G = np.exp(phi)
# # 	G = G / G.sum(axis = 1)[:,np.newaxis]

# # 	return(G) 

# # def SR_linear_ranks(A, beta, alpha = 0):
# # 	np.seterr(divide='ignore', invalid='ignore') 
# # 	s = SpringRank.SpringRank(A.T, alpha = alpha)
# # 	gamma = softmax(s, beta)
# # 	return(gamma)

# # def SR_linear_dynamics(A, beta, alpha = 0):
# # 	'''
# # 	corresponds to the v1 update studied in the paper: each node i has the same probability of endorsing a given node j. That is, attractiveness is a property of endorsed node only. 
# # 	'''
# # 	n = A.shape[0]
# # 	np.seterr(divide='ignore', invalid='ignore') 
# # 	s = SpringRank.SpringRank(A.T, alpha = alpha)
# # 	gamma = softmax(s, beta)
# # 	G = np.tile(gamma, (n,1))
# # 	return(G)

# # def uniform_dynamics(A):
# # 	return np.ones_like(A) / A.shape[0]
