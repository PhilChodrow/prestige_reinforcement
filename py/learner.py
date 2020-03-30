import numpy as np
from py import estimation
from py import simulation
from scipy.optimize import minimize
from numdifftools import Hessian
import warnings

class learner:
	'''
	an object-oriented approach to inference
	'''

	def __init__(self, T = None, A0 = None):

		self.set_data(T, A0)

	def set_data(self, T, A0):
		self.T = T
		self.A0 = A0
		if self.T is not None:
			self.n_rounds = T.shape[0]
			self.n = T.shape[1]

	# -------------------------------------------------------------------------
	# DATA --> FEATURES
	# -------------------------------------------------------------------------

	def set_features(self, feature_list):
		
		self.phi = feature_list
		self.k_features = len(self.phi)

	def set_score(self, score_function):
		'''
		score_function is assumed to return a vector for each matrix
		'''
		self.score = score_function
		
	# -------------------------------------------------------------------------
	# SIMULATION
	# -------------------------------------------------------------------------	
	def simulate(self, beta, lam, A0, n_rounds = 1, update = simulation.stochastic_update, **update_kwargs):

		# setup
		n = A0.shape[0]
		T = np.zeros((n_rounds+1, n, n)) 
		T[0] = A0
		A = T.copy()
		PHI = np.zeros((self.k_features, n, n))

		for t in range(1, n_rounds+1):	

			T[t] = T[t-1]

			# compute scores
			s = self.score(A[t-1])

			# compute features
			for k in range(self.k_features):
				PHI[k] = self.phi[k](s)
			
			# compute rate matrix from scores
			p = np.tensordot(beta, PHI, axes = (0,0))
			GAMMA = np.exp(p)
			GAMMA = GAMMA / GAMMA.sum(axis = 1)[:,np.newaxis]

			# compute update from rate matrix
			Delta = update(GAMMA, **update_kwargs)

			T[t] += Delta
			A[t] += lam*A[t-1] + (1-lam)*Delta	

		return(T)


	# -------------------------------------------------------------------------
	# INFERENCE: FEATURES + PARAMS --> RATES
	# -------------------------------------------------------------------------

	def compute_state_matrix(self, lam):
		self.A = estimation.state_matrix(self.T, lam = lam, A0 = self.A0)

	def compute_score(self):
		'''
		should compute a list of score vectors, one in each timestep
		'''
		S = np.zeros((self.n_rounds, self.n))
		
		for t in range(self.n_rounds):
			S[t] = self.score(self.A[t])

		self.S = S

	def compute_features(self):

		PHI = np.zeros((self.n_rounds, self.k_features, self.n, self.n))
		for t in range(self.n_rounds):
			for k in range(self.k_features):
				PHI[t, k] = self.phi[k](self.S[t])

		self.PHI = PHI

	def compute_rate_matrix(self, beta):

		assert len(beta) == self.k_features

		# GAMMA = np.zeros((self.n_rounds, self.n, self.n))
		
		p = np.tensordot(beta, self.PHI, axes = (0,1))
		GAMMA = np.exp(p)
		GAMMA = GAMMA / GAMMA.sum(axis = 2)[:,:,np.newaxis]

		self.GAMMA = GAMMA


	def ll(self, beta):
		'''
		so what we'd like to do here is separate graph computations from parameter optimization in a principled way. The point is that, having computed the features, the optimization over beta is convex and hopefully fast. 

		features is a hyperarray with axes time x n_features x i x j
		beta is a vector. 

		'''
		
		warnings.filterwarnings("ignore", category=RuntimeWarning)

		self.compute_rate_matrix(beta)
		DT = np.diff(self.T, axis = 0)
		ll = (DT*np.log(self.GAMMA[:-1])).sum()
		return(ll)

	def ML_pars(self, b0 = None, bounds = None):
		
		if b0 is None:
			b0 = np.zeros(self.k_features)

		res = minimize(
			fun = lambda b: -self.ll(b),
			x0 = b0,
			bounds = bounds
			)
		return(res)

	def ML(self, lam0 = .5, alpha0 = 10**(-4), delta = 10**(-4), tol = 10**(-3), step_cap = 0.2, print_updates = False, **kwargs):
		'''
		**kwargs are passed to the optimization over lambda. 
		It's not necessary to control the optimization over the params 
		because the objective is convex. 
		'''
		self.b0 = np.zeros(self.k_features)
		
		def obj(lam):

			self.compute_state_matrix(lam)
			self.compute_score()
			self.compute_features()

			res = self.ML_pars(b0 = self.b0)	
			out = res['fun']
			self.b0 = res['x']
			return(out)

		# bespoke numerical gradient descent with adaptive 
		# learning rate
		print('computing memory hyperparameter lambda')

		lam = lam0
		obj_old = np.inf
		obj_current = obj(lam0)
		# obj_best = np.inf

		alpha = alpha0

		while (obj_old - obj_current > tol):
			
			obj_old = obj_current
			
			alpha = alpha0
			
			deriv = (obj(lam + delta) - obj_current)/delta
			deriv = np.sign(deriv)*np.min((np.abs(deriv), step_cap/alpha))
			
			obj_proposal = np.inf
			proposal = lam - alpha*deriv
			
			while obj_proposal > obj_current:
				
				proposal = lam - alpha*deriv
				obj_proposal = obj(proposal)
				alpha = alpha/2
				
			lam = proposal
			obj_current = obj(proposal)

			if print_updates:
				print('Lambda = ' + str(lam) + ', LL = ' + str(obj_current))
		

		print('computing parameter vector beta')
		self.compute_state_matrix(lam)
		self.compute_score()
		self.compute_features()

		res = self.ML_pars(b0 = self.b0)

		beta = res['x']
		
		return({
			'lam' : lam,
			# 'lam_stderr' : lam_stderr,
			'beta' : beta,
			'LL' : - res['fun']
			}) 	

	def estimate_hessian(self, lam, beta):

		def f(par_vec):

			self.compute_state_matrix(par_vec[0])
			self.compute_score()
			self.compute_features()

			return(self.ll(par_vec[1:]))
		
		x = np.concatenate((np.array([lam]), beta))	
		return Hessian(f)(x)	


	def likelihood_surface(self, LAM, BETA):
		'''
		only implemented for a surface over lambda and a single parameter vector, will error in higher-dimensional models
		'''
		
		lam_grid = len(LAM)
		b_grid = len(BETA)
		
		M = np.zeros((lam_grid, b_grid))

		for i in range(lam_grid):
			lam = LAM[i]
			self.compute_state_matrix(lam = lam)
			self.compute_score()
			self.compute_features()

			for j in range(b_grid):
				beta = BETA[j]
				M[i,j] = self.ll(np.array([beta]))
		
		return(M)









