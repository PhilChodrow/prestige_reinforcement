import numpy as np
from scipy.optimize import minimize
from numdifftools import Hessian
import warnings
from numba import jit


# --------
# update steps 
# --------

# @jit(nopython=True)
def stochastic_update(GAMMA, m_updates_per = 1, m_updates = None):
	'''
	compute a stochastic update from a rate matrix GAMMA, with specified number of updates (samples from the rows of GAMMA). By default, performs a total of n updates, one for each agent. To specify a number of updates, set m_updates. 
	'''
	n = GAMMA.shape[0]
	Delta = np.zeros_like(GAMMA) # initialize

	if m_updates is not None:
		i = np.random.randint(n)
		j = np.random.choice(n, p = GAMMA[i])
		Delta[i,j] += 1	
	else:
		for i in range(n):
			J = np.random.choice(n, p = GAMMA[i], size = m_updates_per)
			Delta[i,J] += 1	
	return(Delta)

@jit(nopython=True)
def deterministic_update(GAMMA, m_updates):
	'''
	compute a *deterministic* update from the rate matrix GAMMA. the deterministic update consists in adding GAMMA itself to the current state, normalized by m_updates/n. 
	'''
	n = GAMMA.shape[0]
	Delta = GAMMA * m_updates / n
	return(Delta)


def state_matrix(T, lam, A0 = None):
	'''
	compute the state matrix for a sequence of updates T using specified memory parameter lam and initial condition A0. 
	'''
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

class model:
	'''
	This class implements an object-oriented approach to simulation and inference. 
	
	The set_data() method is used to provide an initial state matrix A0 and sequence of updates T. The ML() method is then used to perform maximum likelihood inference for the memory and bias parameters. 

	Alternatively, one can simulate from the model by not setting data, and then using the simulate() method with specified parameters. 

	In both cases, it is necessary to set a score function (set_score()) and one or more feature maps (set_features()).


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
		'''
		Each element of feature_list is a function whose argument is a vector of length n and whose return value is an nxn matrix. 
		'''
		self.phi = feature_list
		self.k_features = len(self.phi)


	def set_score(self, score_function):
		'''
		The score function has a single argument, an nxn matrix, and returns a vector of length n. 
		'''
		self.score = score_function
		
	# -------------------------------------------------------------------------
	# SIMULATION
	# -------------------------------------------------------------------------	
	
	
	def simulate(self, beta, lam, A0, n_rounds = 1, update = stochastic_update, align = True, **update_kwargs):
		'''
		Simulate from the model. 
		beta: a 1d np.array() of bias parameters, of the same length as the feature_list argument in self.set_features(). 
		lam: the memory parameter lambda 
		A0: 2d np.array(), the initial state matrix
		n_rounds: number of rounds over which to to perform simulation
		update: update method, one of either stochastic_update or deterministic_update
		align: if True, will attempt to align score vectors. Used when the sign of the score vector is not meaningful; e.g. in the case of the Fiedler vector score. 
		update_kwargs: additional argumentas passed to the function specified in update. 
		'''
		# setup
		n = A0.shape[0]
		T = np.zeros((n_rounds+1, n, n)) 
		T[0] = A0
		A = T.copy()
		PHI = np.zeros((self.k_features, n, n))

		
		self.GAMMA = np.zeros((n_rounds, n, n))
		self.A = np.zeros((n_rounds, n, n))
		self.S = np.zeros((n_rounds, n))

		for t in range(1, n_rounds+1):	

			T[t] = T[t-1]

			# compute scores
			s = self.score(A[t-1])
			if align:
				s_ = self.S[t-2]
				if np.dot(s, s_) < 0:
					s = -s
			self.S[t-1] = s

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
			
			self.GAMMA[t-1] = GAMMA
			self.A[t-1] = A[t-1]
			
		return(T)


	# -------------------------------------------------------------------------
	# INFERENCE: FEATURES + PARAMS --> RATES
	# -------------------------------------------------------------------------

	
	def compute_state_matrix(self, lam):
		self.A = state_matrix(self.T, lam = lam, A0 = self.A0)

	
	def compute_score(self, align = True):
		'''
		should compute a list of score vectors, one in each timestep
		'''
		S = np.zeros((self.n_rounds, self.n))
		
		for t in range(self.n_rounds):
			s = self.score(self.A[t])
			if align:
				if t > 0:
					s_ = S[t-1]
					if np.dot(s, s_) < 0:
						s = -s
			S[t] = s

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

	def ML(self, lam0 = .5, alpha0 = 10**(-4), delta = 10**(-4), tol = 10**(-3), step_cap = 0.2, print_updates = False, align = True, **kwargs):
		'''
		**kwargs are passed to the optimization over lambda. 
		It's not necessary to control the optimization over the params 
		because the objective is convex. 
		'''
		self.b0 = np.zeros(self.k_features)
		
		def obj(lam):

			self.compute_state_matrix(lam)
			self.compute_score(align)
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
		'''
		Estimate the Hessian matrix of the log-likelihood at the maximum likelihood parameter value. Used for estimating error bars on the returned parameters. 
		'''
		def f(par_vec):

			self.compute_state_matrix(par_vec[0])
			self.compute_score()
			self.compute_features()

			return(self.ll(par_vec[1:]))
		
		x = np.concatenate((np.array([lam]), beta))	
		return Hessian(f)(x)	

	def get_rates(self):
		'''
		Return the sequence of rate matrices resulting from either simulation or inference. 
		'''
		return self.GAMMA
	
	def get_scores(self):
		'''
		Return the sequence of scores resulting from either simulation or inference. 
		'''
		return self.S

	def get_states(self):
		'''
		Return the sequence of state matrices resulting from either simulation or inference. 
		'''
		return self.A
	
	def likelihood_surface(self, LAM, BETA):
		'''
		Compute the two-dimensional likelihood surface over the memory parameter lambda a single bias parameter beta. 
		Will error if the model has more than one bias parameter. 
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

