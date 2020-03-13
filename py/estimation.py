import numpy as np
from SpringRank import SpringRank
from scipy.special import gammaln # gammaln(x+1) = log x!

def compute_gamma(A, beta):
    
    # ignore some warning messages surfaced by SpringRank
    np.seterr(divide='ignore', invalid='ignore') 
    
    # compute springranks
    phi = SpringRank.SpringRank(A, alpha = 0)
    
    # exponentiate
    gamma = np.exp(beta * phi)
    
    # normalize
    gamma = gamma / gamma.sum()
    
    return(gamma)    

# construct the increment to add to the current state. Either stochastic or deterministic. def increment(n, gamma, method = 'stochastic'):    

def increment(n, gamma, method = 'stochastic'):
    if method == 'stochastic':
        j = np.random.randint(n)           # uniformly random department gets to hire
        i = np.random.choice(n, p = gamma) # chooses from departments proportional to $\gamma$. 
        E = np.zeros((n,n))
        E[i,j] = 1
        return(E)
    
    elif method == 'deterministic': 
        G = np.tile(gamma, (n,1)).T        # G is the expectation of E above 

        return(G)


def synthetic_time_series(n, beta, lam, n_rounds, T0):
    
    T0 = T0
#     T0 = np.random.randint(10, 20, size = (n,n))
#     T0 = T0 / T0.sum()

    T = np.zeros((n_rounds, n, n))

    A = np.zeros_like(T)

    T[0] = T0
    A[0] = T0

    for j in range(1,n_rounds): 

        T[j] = T[j-1]
        gamma = compute_gamma(A[j-1], beta) 

        num_increment = np.random.randint(n,n**2) # can change
        to_add = np.zeros((n,n))

        for k in range(num_increment):
            to_add += increment(n, gamma, 'stochastic')

        T[j] += to_add

        A[j] = lam*A[j-1] + (1-lam)*to_add
        
    return(T)

def state_matrix(T, lam, A0 = None):
    n_rounds = T.shape[0]
    DT = np.diff(T, axis = 0)
    A = np.zeros_like(T)
    if A0 is None:
        A[0] = T[0]
    else:
        A[0] = A0
    for j in range(1,n_rounds):
        A[j] = lam*A[j-1]+(1-lam)*DT[j-1]
    return(A)

# def nuLL(T, lam, A0 = None, alpha = 0):
    
#     n_rounds, n = T.shape[0], T.shape[1]
        
#     A = state_matrix(T,lam,A0)
#     DT = np.diff(T, axis = 0)
#     increments = DT.sum(axis = 0)

#     D = np.zeros((n_rounds-1,n))
    
#     for i in range(0, n_rounds-1):
#         D[i] = A.sum(axis = 1) # check
    
#     counts = DT.sum(axis = )

def LL(T, lam, A0 = None, fun = SpringRank.SpringRank, **kwargs):
               
    n_rounds, n = T.shape[0], T.shape[1]
        
    A = state_matrix(T,lam,A0)
    DT = np.diff(T, axis = 0)
    increments = DT.sum(axis = 0)
    
    S = np.zeros((n_rounds-1,n))
    
    for i in range(0, n_rounds-1):
        S[i] = fun(A[i], **kwargs)
    
    counts = DT.sum(axis = 2)    
    K = counts.sum(axis = 1)
    
    def ll(beta):
        first = (beta*counts*S).sum()
        second = (K*np.log(np.exp(beta*S).sum(axis = 1))).sum()
        constant = (gammaln(K + 1) - K*np.log(n) - gammaln(counts + 1).sum(axis = 1)).sum()
        
        return(first - second + constant)
        
    return(ll)

def likelihood_surface(T, LAMBDA, BETA, A0 = None, alpha = 0, fun = SpringRank.SpringRank, **kwargs):
    
    n_lambda = len(LAMBDA)
    n_beta = len(BETA)
    
    M = np.zeros((n_lambda, n_beta))
    for l in range(n_lambda):
        ll = LL(T,LAMBDA[l], A0, fun = fun, **kwargs)
        M[l] = np.array([ll(b) for b in BETA])
        
    return({'M' : M, 'LAMBDA' : LAMBDA, 'BETA' : BETA})

def hessian(M, X, Y):
    """
    Based on 
    https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    """
    M_grad = np.gradient(M, X, Y) 
    hessian = np.empty(M.shape + (M.ndim, M.ndim), dtype=M.dtype) 
    for k, grad_k in enumerate(M_grad):
        tmp_grad = np.gradient(grad_k, X, Y) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[:, :,k, l] = grad_kl
    return hessian

def get_estimates(result):
    
    M = result['M']
    BETA = result['BETA']
    LAMBDA = result['LAMBDA']
    
    ix = np.where(M == M.max())
    beta_hat = BETA[ix[1]][0]
    lambda_hat = LAMBDA[ix[0]][0]

    # stderrs from Fisher Information Matrix
    try:
        stderrs = np.sqrt(-np.diag(np.linalg.inv(hessian(M, BETA, LAMBDA)[ix])[0]))

        return({'beta' : beta_hat, 
               'lambda' : lambda_hat,
               's_beta' : stderrs[1] ,
               's_lambda' : stderrs[0]})
    except np.linalg.LinAlgError:
        return({'beta' : beta_hat, 
               'lambda' : lambda_hat,
               's_beta' : np.nan,
               's_lambda' : np.nan})

    
def pageRank(A, alpha = 0.15):
    D_inv = np.diag(1/A.sum(axis = 1))
    A_ = A.dot(D_inv)
    B = (1-alpha)*A_ + alpha*np.ones_like(A)
    eig = np.linalg.eig(B)
    return(np.abs(eig[1][:,0]))


# connectedness issues

def check_connected(A):

    A_ = (A + A.T) / 2
    D = np.diag(A_.sum(axis = 1))
    L = D - A_
    eig = np.linalg.eigh(L)
    if (eig[0] < 10**(-30)).sum() >= 1: 
        return(False)
    else:
        return(True)