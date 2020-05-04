import sys
sys.path.append('py/')

from model import *
from scores import *
from utils import *

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numba import jit

# ---------------------------------------------
# PARAMS
# ---------------------------------------------

# memory parameter

lam = 0.995

# possible values of beta_1 and beta_2
BETA_1 = np.array([1,2,3])
BETA_2 = np.array([-2, -1, 0])

# number of rounds and updates per round
n_rounds = 2000
m_updates = 1

# number of agents
n = 8

# for reproducibility -- change to get a different plot
np.random.seed(seed=6) 

# generate initial condition
A0 = np.random.rand(n,n)
A0 = A0/A0.sum()

# column of traces for which to show the end state. 
highlight_col = 2

# min and max of vertical axis
ymax = 0.5
ymin = 0.0

# path to save file

save_path = 'fig/dynamics_examples.png'


# ---------------------------------------------
# Visualization commands
# ---------------------------------------------

matplotlib.rcParams['font.sans-serif'] = "Arial Unicode MS"

cset = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
cset_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD','#000000']

# ---------------------------------------------
# Feature definitions
# ---------------------------------------------

@jit(nopython=True)
def linear_feature(s):
    return(np.outer(np.ones(len(s)), s))

def quadratic_feature(s):
    S = linear_feature(s)
    V = (S - S.T)**2
    return(V)

# ---------------------------------------------
# Main loop
# ---------------------------------------------

if __name__ == '__main__':

    fig, axarr = plt.subplots(3,4, figsize = (16, 8))

    for i in range(3):
        for j in range(3):
            ax = axarr[j,i]
            beta_1 = BETA_1[i]
            beta_2 = BETA_2[j]

            M = model()
            M.set_score(score_function = lambda A: homebrew_SpringRank_score(A.T))
            M.set_features([linear_feature, 
                            quadratic_feature])

            M.simulate(beta = np.array([beta_1, beta_2]), 
                    lam = lam, 
                    A0 = A0, 
                    n_rounds = n_rounds, 
                    update = stochastic_update, 
                    m_updates = m_updates)
            GAMMA = M.get_rates()
            
            for k in range(n):
                p = ax.plot(GAMMA.mean(axis = 1)[np.int(n_rounds/2):,k], color = cset_muted[k])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set(ylim = (ymin, ymax))
            ax.set(xlim = (0, np.int(n_rounds/2)))
            
            if j < 2:
                ax.spines['bottom'].set_visible(False)
                ax.xaxis.set_ticks([])
                plt.xticks([])

            if i > 0:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks([])
                plt.yticks([])

            if j == 2:
                ax.set(xlabel = r'$t$')
            if i == 0:
                ax.set(ylabel = r'$\gamma$')
            
            ax.set_title(
                r'$\beta_1 = $' + str(round(beta_1,2)) + r'$\;$,$\;$ $\beta_2 = $' + str(round(beta_2,2))
            )
            
            if i == highlight_col:
                ax = axarr[j,3]
                A = M.get_states()[-1]

                v = GAMMA[-1].mean(axis = 0) 
                ax.imshow(np.ones_like(A), cmap = 'Greys', vmax = 1, vmin = 0, alpha = 1)
                ax.imshow(matrix_sort(A, -v), vmax = .1,  cmap = 'Greys', alpha = .99)
                ax.set_ylabel(r'$\longrightarrow$', rotation=0, fontsize=20, labelpad=30, color = 'black')
                for pos in ['bottom', 'top', 'left', 'right']:
                    ax.spines[pos].set_visible(False)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

        axarr[0,3].set_title('Final State (Third Column)')

        plt.subplots_adjust(wspace = 0.1)

        plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)

