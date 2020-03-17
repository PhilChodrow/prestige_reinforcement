import numpy as np
from SpringRank import SpringRank
from scipy.special import gammaln
from py import features

def stochastic_update(GAMMA, m_updates):
	n = GAMMA.shape[0]
	Delta = np.zeros_like(GAMMA) # initialize
	for k in range(m_updates):
		# endorser is uniformly random
		i = np.random.randint(n) 
		# endorsed according to bar_delta[i]
		j = np.random.choice(n, p = GAMMA[i]) 
		Delta[i,j] += 1	

	return(Delta)

def deterministic_update(GAMMA, m_updates):
	n = GAMMA.shape[0]
	Delta = GAMMA * m_updates / n
	return(Delta)




