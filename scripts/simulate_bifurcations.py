import sys
sys.path.append('py/')

from model import *
from scores import *

import numpy as np
from numba import jit

scores = [
    lambda A: homebrew_SpringRank_score(A.T),
    lambda A: powered_degree_score(A, .5),
    lambda A: eigenvector_score(A.T)
]

labels = ['SpringRank', 'Root_Degree', 'Eigenvector']

b_grid = 21

BETA_VECS = [
    np.linspace(0, 10, b_grid),
    np.linspace(0, 10, b_grid),
    np.linspace(0, 20, b_grid)
]

lam = .9995
n_rounds = 50000

n = 10

@jit(nopython=True)
def linear_feature(s):
    return(np.outer(np.ones(len(s)), s))

if __name__ == '__main__':

    A0 = np.random.rand(n,n)
    A0 = A0/A0.sum()

    for i in range(3):

        print('Simulating dynamics with score function ' + labels[i])

        BETAS = BETA_VECS[i]
        
        M = model()
        M.set_score(score_function = scores[i])
        M.set_features([linear_feature])
        
        V = np.zeros((b_grid, n))
        
        for j in range(b_grid):
            M.simulate(beta = np.array([BETAS[j]]), 
                            lam = lam, 
                            A0 = A0, 
                            n_rounds = n_rounds, 
                            update = stochastic_update, 
                            m_updates = 1)
            
            GAMMA = M.get_rates()
            GAMMA = np.sort(GAMMA, axis = 2)
            V[j] = GAMMA[(-5000):(-1)].mean(axis = (0,1))

        V = np.concatenate((BETAS[:,np.newaxis], V), axis = 1)

        save_path = 'throughput/' + labels[i] + '_bifurcation.txt'
        np.savetxt(save_path, V)