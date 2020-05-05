
import numpy as np
import pandas as pd
from itertools import product

import matplotlib
from matplotlib import pyplot as plt

# MATH PARAMS
n = 10

labels = ['SpringRank', 'Root-Degree', 'Eigenvector']

instabilities = [2, 2*n**(1/2), 3*n**(1/2)]
emergence = [np.nan, np.nan, n**(1/2)]

# VIZ PARAMS
matplotlib.rcParams['font.sans-serif'] = "Arial Unicode MS"
cset = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
cset_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD','#000000']

# read and clean dfs containing the curves

def read_SpringRank_curves():
    df_SR = pd.read_csv('numerics/data_n_10.csv') # springRank
    df_SR['s_1'] = df_SR.s1
    df_SR['s_2'] = df_SR.s2
    df_SR['n_1'] = df_SR.n1
    df_SR['group'] = 1
    df_SR = df_SR.drop(['s1', 's2', 'n2', 'n1'], axis = 1)
    df_SR['stable'] = True
    # need to pad this one from zero to 2

    padding = pd.DataFrame(
        list(product(np.linspace(0,2, 21),
        [0], 
        [0], 
        [1], 
        [1], 
        [True])),
        columns = ['beta', 's_1', 's_2', 'n_1', 'group', 'stable']
    )
    df_SR = pd.concat((padding, df_SR), axis = 0)

    # need to pad to add unstable egalitarian solution

    padding = pd.DataFrame(
        list(product(np.linspace(2,10, 21),
        [0], 
        [0], 
        [1], 
        [1], 
        [False])),
        columns = ['beta', 's_1', 's_2', 'n_1', 'group', 'stable']
    )

    df_SR = pd.concat((padding, df_SR), axis = 0)

    return(df_SR)

def read_degree_curves():
    return(pd.read_csv('throughput/degree_score_curves.csv'))

def read_EV_curves():
    return(pd.read_csv('throughput/eigenvector_score_curves.csv'))

# computation functions

def compute_gamma(row): 
    beta = row.beta
    s_1 = row.s_1
    s_2 = row.s_2
    n_1 = row.n_1
    n_2 = n - n_1
    
    v_1 = np.exp(beta*s_1)
    v_2 = np.exp(beta*s_2)
    
    gamma_1 = np.array(v_1/(n_1*v_1 + n_2*v_2))
    gamma_2 = np.array(v_2/(n_1*v_1 + n_2*v_2))
    
    return(pd.Series([gamma_1, gamma_2], 
                     index = ['gamma_1', 'gamma_2']))

if __name__ == '__main__':

    df_SR = read_SpringRank_curves()
    df_deg = read_degree_curves()
    df_EV = read_EV_curves()

    df_SR = pd.concat((df_SR, df_SR.apply(compute_gamma, result_type = 'expand', axis = 1)), axis = 1)
    df_deg = pd.concat((df_deg, df_deg.apply(compute_gamma, result_type = 'expand', axis = 1)), axis = 1)
    df_EV = pd.concat((df_EV, df_EV.apply(compute_gamma, result_type = 'expand', axis = 1)), axis = 1)

    ix = df_deg.gamma_1 > .9
    df_deg.stable[ix] = True

    sim_SR = np.loadtxt('throughput/SpringRank_bifurcation.txt')
    sim_deg = np.loadtxt('throughput/Root_Degree_bifurcation.txt')
    sim_EV = np.loadtxt('throughput/Eigenvector_bifurcation.txt')

    dfs = [df_SR, df_deg, df_EV]
    sims = [sim_SR, sim_deg, sim_EV]

    fig, axarr = plt.subplots(1,3, figsize = (15, 3.5))

    for i in range(3):
        ax = axarr[i]
        df = dfs[i]
        V = sims[i]
        BETAS = V[:,0]
        V = V[:,1:]
        # background curves   

        for gamma in ['gamma_1', 'gamma_2']:
            p = df.groupby(['n_1','group', 'stable']).apply(
                lambda g: ax.plot(g.beta, 
                                  g[gamma], 
                                  zorder = 1,
                                  color = 'lightgrey', 
                                  linewidth = 1)
                )
            df_sub = df[df.stable]
            p = df_sub.groupby(['n_1','group', 'stable']).apply(
                lambda g: ax.plot(g.beta, 
                                  g[gamma], 
                                  zorder = 1,
                                  color = 'black', 
                                  linewidth = 1)
                )
        
        for k in range(V.shape[1]):
            
            p = ax.scatter(BETAS, V[:,k], alpha = 1, s=20, zorder = 2, facecolors='none', edgecolors = cset[0])

        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set(xlabel = r'$\beta$')
        ax.set(ylabel = r'$\gamma$')

    plt.savefig('fig/bifurcations.png', dpi = 300, bbox_inches = 'tight')







