import numpy as np
from scipy.optimize import minimize
from numdifftools import Hessian
import warnings
from numba import jit
from scipy.special import gammaln


################################################################################
# UPDATE FUNCTIONS
################################################################################

# @jit(nopython=True)
def stochastic_update(GAMMA, m_updates_per = 1, m_updates = None):
	'''
	compute a stochastic update from a rate matrix GAMMA, with specified number of updates (samples from the rows of GAMMA). By default, performs a total of n updates, one for each agent. To specify a number of updates, set m_updates. 
	GAMMA: np.array(), a square matrix whose ijth entry gives the probability that agent i endorses agent j, conditional on agent i being selected to make an endorsement. 
	m_updates_per: int, the number of endorsements per agent. If m_updates per is set and m_updates is not, then each agent will make m_updates_per endorsements per timestep. 
	m_updates: int, the number of total endorsements. If m_updates is not None, then an agent will be selected uniformly at random and will make an endorsement, with this process being repeated m_updates times. 
	'''
	n = GAMMA.shape[0]
	Delta = np.zeros_like(GAMMA) # initialize

	# if m_updates is set, randomly select agents to make endorsements a total of m_updates times.
	if m_updates is not None: 
		i = np.random.randint(n)
		j = np.random.choice(n, p = GAMMA[i])
		Delta[i,j] += 1	
	# otherwise, each agent makes m_updates_per endorsements. 
	else: 
		for i in range(n):
			J = np.random.choice(n, p = GAMMA[i], size = m_updates_per)
			Delta[i,J] += 1	
	return(Delta)

@jit(nopython=True)
def deterministic_update(GAMMA, m_updates):
	'''
	compute a *deterministic* update from the rate matrix GAMMA. the deterministic update consists in adding GAMMA itself to the current state, normalized by m_updates/n. 	
	Multiply m_updates by n to obtain a deterministic analog of m_updates_per. 
	GAMMA: np.array(), a square matrix whose ijth entry gives the probability that agent i endorses agent j, conditional on agent i being selected to make an endorsement. 
	m_updates: int, the number of total updates to make. 
	'''
	n = GAMMA.shape[0]
	Delta = GAMMA * m_updates / n
	return(Delta)

################################################################################
# STATE MATRIX FROM DATA
################################################################################

def state_matrix(T, lam, A0 = None):
	'''
	compute the state matrix for a sequence of cumulative data T using specified memory parameter lam and initial condition A0. 
	T: np.array(), a 3-dimensional array in which the first axis indexes time. T[t] is an n x n matrix in which the ijth entry gives the number of endorsements of j by i by timestep t.  
	lam: float, a memory parameter between zero and one. 
	A0: np.array(), a 2-dimensional array specifying the state matrix at system initialization. 
	'''
	n_rounds = T.shape[0]
	
	# extract the incremental updates associated with T
	DT = np.diff(T, axis = 0)

	# pre-allocate the state matrix
	A = np.zeros_like(T).astype(float)
	
	# if initial state is not specified, then just use first layer of T[0]
	if A0 is None:
		A[0] = T[0]
	else:
		A[0] = A0
	
	# construct state matrix according to update equation
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

################################################################################
# SET MODEL ATTRIBUTES
################################################################################

	def set_features(self, feature_list):
		'''
		Set the feature maps associated with the model. 
		feature_list: a list() of functions from n-vectors to n x n matrices. 
		'''
		self.phi = feature_list
		self.k_features = len(self.phi)


	def set_score(self, score_function):
		'''
		Set the score function of the model. 
		The score function take a single argument, an n x n matrix, and returns a vector of length n. 
		'''
		self.score = score_function
		
################################################################################
# SIMULATION
################################################################################
	
	def simulate(self, beta, lam, A0, n_rounds = 1, update = stochastic_update, align = True, **update_kwargs):
		'''
		Simulate from the model. 
		beta: a 1d np.array() of bias parameters, of the same length as the feature_list argument in self.set_features(). 
		lam: the memory parameter lambda 
		A0: 2d np.array(), the initial state matrix
		n_rounds: number of rounds over which to to perform simulation
		update: update method, one of either stochastic_update or deterministic_update
		align: if True, will attempt to align score vectors. Used when the sign of the score vector is not meaningful; e.g. in the case of the Fiedler vector score. 
		update_kwargs: additional arguments passed to the function specified in update. 
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

################################################################################
# INFERENCE HELPERS
################################################################################

	def compute_state_matrix(self, lam):
		"""
		Compute the sequence of state matrices for specified memory parameter lam. 
		"""
		self.A = state_matrix(self.T, lam = lam, A0 = self.A0)

	
	def compute_score(self):
		'''
		Compute the sequence of score vectors (one in each timestep), using self.score(). Requires that self.compute_state_matrix() has already been called. 
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
		"""
		Compute the array of feature matrices PHI. Requires that self.compute_score() has already been called. 
		"""
		PHI = np.zeros((self.n_rounds, self.k_features, self.n, self.n))
		for t in range(self.n_rounds):
			for k in range(self.k_features):
				PHI[t, k] = self.phi[k](self.S[t])

		self.PHI = PHI

	def compute_rate_matrix(self, beta):
		"""
		Compute the rate matrix GAMMA for specified preference parameters beta. 
		Requires that self.compute_scores() and self.compute_features() have already been called. 
		beta: np.array(), the preference parameter
		"""
		assert len(beta) == self.k_features

		# GAMMA = np.zeros((self.n_rounds, self.n, self.n))
		
		p = np.tensordot(beta, self.PHI, axes = (0,1))
		GAMMA = np.exp(p)
		GAMMA = GAMMA / GAMMA.sum(axis = 2)[:,:,np.newaxis]

		self.GAMMA = GAMMA

################################################################################
# INFERENCE: OBJECTIVES AND ALGORITHMS
################################################################################

	def ll(self, beta):
		'''
		Objective function for likelihood maximization as a function of beta, with lambda implicitly fixed. Requires that self.compute_score() and self.compute_features() have already been run. 	
		beta: np.array(), the preference parameter. 
		'''
		
		warnings.filterwarnings("ignore", category=RuntimeWarning)

		self.compute_rate_matrix(beta)
		DT = np.diff(self.T, axis = 0)
		C = gammaln(DT.sum(axis = 1)+1).sum() - gammaln(DT+1).sum()
		ll = (DT*np.log(self.GAMMA[:-1])).sum() + C

		return(ll)

	def ML_pars(self, b0 = None):
		"""
		Estimate the preference parameter from data using scipy.minimize(). 
		This optimization is convex and requires no control parameters. 
		b0: np.array(), the initial guess for the preference parameter. Generally unnecessary to set in practical settings. 
		"""
		
		if b0 is None:
			b0 = np.zeros(self.k_features)

		res = minimize(
			fun = lambda b: -self.ll(b),
			x0 = b0
			)
		return(res)

	def ML(self, lam0 = .5, alpha0 = 10**(-4), delta = 10**(-4), tol = 10**(-3), step_cap = 0.2, print_updates = False, align = True):
		'''
		Estimate the parameters lambda and beta from empirical data. 
		Requires that self.set_data() has already been called. 
		Because the problem is convex in beta, we use ML_pars (which in turn calls scipy.minimize()) to evaluate the objective as a function of lambda. 
		We then optimize this objective via hill-climbing. 
		lam0: float, the initial guess for lambda
		alpha0: float, the initial step size allowed when performing hill-climbing over lambda
		delta: float, the increment used to estimate the derivative of the likelihood with respect to lambda using forward differences. 
		tol: float. If the objective changes by less than tol in an interation, the algorithm is considered to have converged. 
		step_cap: float, the largest step size allowable when performing hill-climbing over lambda. 
		print_updates: bool, if true, prints updates after each outer iteration. 
		align: passed to self.compute_score()
		'''

		# initial guess for beta
		self.b0 = np.zeros(self.k_features)
		
		# objective function in lambda, obtained by optimizing over beta using self.ML_pars(). Each call sets self.b0 to the obtained value of beta to encourage faster convergence. 

		def obj(lam):
			self.compute_state_matrix(lam)
			self.compute_score(align)
			self.compute_features()

			res = self.ML_pars(b0 = self.b0)	
			out = res['fun']
			self.b0 = res['x']
			return(out)

		# bespoke numerical gradient ascent with adaptive 
		# learning rate for learning lambda
		print('computing memory hyperparameter lambda')

		# initialize
		lam = lam0
		obj_old = np.inf
		obj_current = obj(lam0)
		alpha = alpha0

		while (obj_old - obj_current > tol):
			
			obj_old = obj_current
			alpha = alpha0
			
			# estimate the derivative of the objective via forward differences, thresholding against the control parameters. Computed only once per outer iteration; controls the initial step. 

			deriv = (obj(lam + delta) - obj_current)/delta
			deriv = np.sign(deriv)*np.min((np.abs(deriv), step_cap/alpha))
			
			obj_proposal = np.inf
			proposal = lam - alpha*deriv
			
			# keep trying progressively smaller steps in the indicated direction until one of them improves the objective
			while obj_proposal > obj_current:
			
				proposal = lam - alpha*deriv
				obj_proposal = obj(proposal)
				alpha = alpha/2
				
			lam = proposal
			obj_current = obj(proposal)

			if print_updates:
				print('Lambda = ' + str(lam) + ', LL = ' + str(obj_current))
		
		# after the optimal lambda has been identified, perform one last optimization over beta to obtain the estimated preference parameter. 
		print('computing parameter vector beta')
		self.compute_state_matrix(lam)
		self.compute_score()
		self.compute_features()

		res = self.ML_pars(b0 = self.b0)

		beta = res['x']
		
		return({
			'lam' : lam,
			'beta' : beta,
			'LL' : - res['fun']
			}) 	

	def estimate_hessian(self, lam, beta):
		'''
		Estimate the Hessian matrix of the log-likelihood at the maximum likelihood parameter value. Used for estimating error bars on the returned parameters. 
		lam: float, the estimated value of lambda
		beta: np.array(), the estimated value of the preference parameter vector beta. 
		'''
		def f(par_vec):

			self.compute_state_matrix(par_vec[0])
			self.compute_score()
			self.compute_features()

			return(self.ll(par_vec[1:]))
		
		x = np.concatenate((np.array([lam]), beta))	
		return Hessian(f)(x)	

################################################################################
# GETTERS FOR INSPECTION + VISUALIZATION
################################################################################

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

################################################################################
# LIKELIHOOD SURFACE FOR VISUALIZATION
################################################################################

	def likelihood_surface(self, LAM, BETA):
		'''
		Compute the two-dimensional likelihood surface over the memory parameter lambda and a single bias parameter beta. 
		Will error if the model has more than one bias parameter. 
		LAM: 1-d np.array(), the set of values of lambda over which to compute the likelihood
		BETA: 1-d np.array(), the set of values of beta over which to compute the likelihood
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

