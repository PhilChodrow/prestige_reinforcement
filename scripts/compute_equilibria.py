
import numpy as np
import pandas as pd
from itertools import product 
from scipy.optimize import root

'''
What we want to do here is to write a *script* that will reproduce the curves computed by degree_numerics and eigenvector_numerics. So, needs to pull on those two notebooks, and consolidate where possible to minimize duplication. Can eventually fold in SpringRank computations as well, if we can work with Mari. 
'''


def compute_gamma(s, beta):
    gamma = np.exp(beta*s)
    gamma = gamma/gamma.sum()
    return(gamma)

def make_s(s_1, s_2, n_1, n):
        
    s_1 = s_1
    s_2 = s_2
    
    n_1 = int(n_1)
    
    s = np.zeros(n)
    s[0:n_1] += 1*s_1
    s[n_1:n] += 1*s_2

    return(s)

def compute_degree_equilibria():

    # constants
    BETA_GRID = np.linspace(0, 10, 201)
    n_1   = np.arange(1, 6)
    rep = np.arange(0, 101)
    n = 10

    def g(row):
        x0 = np.random.rand(2)
        x0 = x0 / x0.sum()
        res = root(lambda s: f(s[0],s[1], row.n_1, n, row.beta)[[0,n-1],], x0 = x0, tol = 10**(-8))
        if res['success']:
            return(pd.Series(res['x'], index = ['s_1', 's_2']))
        else:
            return(pd.Series([np.nan, np.nan], index = ['s_1', 's_2']))

    def make_gamma(s_1, s_2, n_1, n, beta):
        s = make_s(s_1, s_2, n_1, n)

        return(compute_gamma(np.sqrt(s), beta)) # for root-degree score

    def f(s_1, s_2, n_1, n, beta):
        return(make_s(s_1, s_2, n_1, n) - make_gamma(s_1, s_2, n_1, n, beta))

    def jacobian(n_1, s_1, s_2, n, beta):
        s = make_s(s_1, s_2, n_1, n)
        gamma = compute_gamma(s, beta)
        Gamma = np.diag(gamma)
        J = beta/2*(Gamma - np.outer(gamma, gamma)).dot(np.diag(s**(-1/2)))
        return(J)

    def test_stable(n_1, s_1, s_2, n, beta):
        J = jacobian(n_1, s_1, s_2, n, beta)
        try:
            stable = np.abs(np.linalg.eig(J)[0]).max() < .98 # bit of a fudge factor for funky numerics
            return(stable)
        except np.linalg.LinAlgError:
            return(False)

    # initialize empty data frame

    df = pd.DataFrame(list(product(BETA_GRID, n_1, rep)), columns=['beta', 'n_1', 'rep'])

    # main loop
    
    # compute a bunch of equilibria
    df = pd.concat([df, df.apply(g, axis = 1, result_type='expand')], axis = 1) 
    
    # organization, bookkeeping, smoothing
    df = df.sort_values(['beta', 'n_1', 's_1'])
    df = df[df.s_1.notnull()]
    df['s_1'] = np.round(df.s_1, 3)
    df['s_2'] = np.round(df.s_2, 3)
    df['s1'] = df[['s_1', 's_2']].max(axis = 1)
    df['s2'] = df[['s_1', 's_2']].min(axis = 1)
    df = df.drop_duplicates(['n_1', 'beta', 's1', 's2'])
    df = df.drop(['s_1', 's_2', 'rep'], axis = 1)

    df['group'] = df.groupby(['beta', 'n_1'])['s1'].rank(method = 'first')

    # get values for s 
    df = pd.concat([df, df.apply(lambda row: pd.Series(make_s(row.s1, row.s2, row.n_1, n)[[0,-1]], index = ['s_1', 's_2']),
                                result_type = 'expand',
                                axis = 1)],
                axis = 1)
    df = df.drop(['s1','s2'], axis = 1)

    # test for stability
    df['stable'] = df.apply(lambda row: test_stable(row.n_1, row.s_1, row.s_2, n, row.beta), axis = 1)

    df.to_csv('throughput/degree_score_curves.csv', index = False)

def compute_eigenvector_equilibria():
    def f(s_1, s_2, n_1, n, beta, max_iters = 200, tol = 10**(-5)):
        n = int(n)
        s = make_s(s_1, s_2, n_1, n)
        gamma = compute_gamma(s, beta)
        
        for i in range(max_iters):
            s_old = s.copy()
            gamma = compute_gamma(s, beta)
            G = np.tile(gamma, (n,1))
            eigs = np.linalg.eig(G.T)
            which_eig = np.argmax(np.abs(eigs[0]))
            v = np.abs(eigs[1][:,which_eig])
            s = v / np.sqrt((v**2).sum())
            s = -np.sort(-s)
            
            # smoothing: 
            for i in range(n):
                for j in range(n):
                    if np.abs(s[i] - s[j]) < 10**(-4):
                        s[j] = s[i]
            
            if np.sqrt(((s - s_old)**2).sum()) < tol:
                return(
                    pd.Series([s[0], s[-1]], index = ['s_1', 's_2'])
                )
        return(
            pd.Series([s[0], s[-1]], index = ['s_1', 's_2'])
        )
    
    def g(row):
        return(f(row.s_1_0, row.s_2_0, row.n_1, row.n, row.beta, max_iters = 5000, tol = 10**(-8)))     


    BETA_GRID = np.linspace(0, 20, 501)
    n_1 = np.arange(1, 6)
    s1 = [1.1, 1.000001]
    s2 = [1]

    n = 10

    df = pd.DataFrame(list(product(BETA_GRID, n_1, s1, s2)), columns=['beta', 'n_1', 's_1_0', 's_2_0'])

    df['n'] = n

    df = pd.concat([df, df.apply(lambda row: g(row), 
                                axis = 1, 
                                result_type='expand')], 
                axis = 1) # compute a bunch of equilibria

    df = df.sort_values(['beta', 'n_1', 's_1'])
    df = df[df.s_1.notnull()]
    df['s_1'] = np.round(df.s_1, 3)
    df['s_2'] = np.round(df.s_2, 3)
    df = df.drop_duplicates(['n_1', 'beta', 's_1', 's_2'])
    df['group'] = df.groupby(['beta', 'n_1'])['s_1'].rank(method = 'first')
    df = df.drop(['s_1_0', 's_2_0', 'n'], axis = 1)

    def numerical_jacobian(n_1, s_1, s_2, n, beta):
    
        def h(s):
            gamma = compute_gamma(s, beta)
            G = np.tile(gamma, (n,1))
            v = G.T.dot(s) - (s.T.dot(G.T).dot(s))*s
            return(v)
        
        J = np.zeros((n,n))
        eps = 10**(-10)
        s = make_s(s_1, s_2, n_1, n)
        for j in range(n):
            s_ = s.copy()
            s__ = s.copy()
            s_[j] += eps
            s__[j] -= eps
            
            J[j,:] = (h(s_) - h(s__))/(2*eps)
        
        return(J)

    def test_stable_(n_1, s_1, s_2, n, beta):
        '''
        A good upgrade to this function would be for it to return values that distinguish between the stability regimes. 
        '''
        J = numerical_jacobian(n_1, s_1, s_2, n, beta)

        eigs = np.linalg.eig(J)

        which_eig = np.argmax(np.abs(eigs[0]))
        vec = eigs[1][:,which_eig]
        if(np.var(vec) < 10**(-3)):
            dominant_egalitarian = True
        else:
            dominant_egalitarian = False
        
        if eigs[0].max() < 0:
            linearly_stable = True
        else:
            linearly_stable = False
        return(dominant_egalitarian or linearly_stable)

    df['stable'] = df.apply(lambda row: test_stable_(row.n_1, row.s_1, row.s_2, n, row.beta), axis = 1)
    df.to_csv('throughput/eigenvector_score_curves.csv', index = False)

if __name__ == '__main__': 
    compute_degree_equilibria()
    compute_eigenvector_equilibria()