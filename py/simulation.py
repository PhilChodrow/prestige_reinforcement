import numpy as np
from SpringRank import SpringRank
from scipy.special import gammaln
from py import features

# So I think the general spec we are looking for is that the user can supply a function which gives an arbitrary *matrix* of probabilities, which can then be used for both forward simulation and backward inference. In the case of backward inference, we might also want to ask the user to supply gradients, but we'll get there.

# in both the SIMULATION and the DATA ANALYSIS, agent i is a uniformly random endorser of agent j. 

def stochastic_update(GAMMA, m_updates_per = 1, m_updates = None):
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

def deterministic_update(GAMMA, m_updates):
	n = GAMMA.shape[0]
	Delta = GAMMA * m_updates / n
	return(Delta)




