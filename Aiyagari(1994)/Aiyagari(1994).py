# -*- coding: utf-8 -*-
"""
Replication of Aiyagari(1994)

@author: Taiki Ono
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import bisect
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, linalg
from Tauchen import tauchen_approximation
import time
from tqdm import tqdm

# 1. Parameter Setup 
beta = 0.96                 # discount factor
mu = 3                      # relative risk aversion (parameter of CRRA utility function)
alpha = 0.36                # capital share
delta = 0.08                # capital depreciation
na = 1000                    # number of grid points of asset
nl = 5                      # number of grid points of labor endowments
rho = 0.3                  # persistence of the shock
sigma = 0.2                 # standard deviation of shock
std = np.sqrt((sigma**2)*(1-rho**2)) # variance of the random term in the model
abar = 0                    # borrowing constaraint
amax = 20                   # maximum value of agrid
L = 1                       # number of workers


# approximating labor endowment and its Markov chain by Tauchen mehod
l, P = tauchen_approximation(rho,std,b=0,m=3,n=nl)  
lgrid = np.exp(l) #convert z into exp because log(l) follows AR(1)
agrid = np.linspace(abar,amax,na)

# 2. build functions for computation 
@njit("f8(f8)")
def CRRA(cons):
    """
    ----------------------------------------------------------------------
    Returns utility level of CRRA utility function given consumption level
    ----------------------------------------------------------------------
    <input>
    ・ cons: consumption level
    <output>
    ・ utility level given consumption level
    """
    if cons>0:
        return cons**(1-mu) / (1-mu)
    else:
        return -100000

def f(K,L):
    """
    ---------------------------------------------------------------------------
    Cobb-Douglas production function
    ---------------------------------------------------------------------------
    <input>
    ・ K : total capital in economy
    ・ L : total labor in economy
    <output>
    ・ output given K and L
    """    
    return (K**alpha)*(L**(1-alpha))

def f_prime(K,L):
   """
   ---------------------------------------------------------------------------
   First-order derivative of production function w.r.t K and L
   ---------------------------------------------------------------------------
   <input>
   ・ K : total capital in economy
   ・ L : total labor in economy
   <output>
   ・ MPK and MPL given K and L
   """
   MPK = alpha*K**(alpha-1) * L**(1-alpha)
   MPL = (1-alpha)*K**alpha * L**(-alpha)
   
   return MPK, MPL

@njit
def utility(mat,w,r):
    """
    ---------------------------------------------------------------------------
    Returns utility matrix given asset of two periods and states 
    ---------------------------------------------------------------------------
    <input>
    ・ mat: empty matrix(na-by-na-by-ns matrix)
    ・ w : wage
    ・ r : interest rate
    <output>
    ・ utility matrix (na-by-na-by-ns matrix)
    where ・ na: number of asset grids(i: this period, j: next period)
          ・ ns: number of states 
    """
    na, nl = len(agrid), len(lgrid)
    
    for i in range(na): #asset this period
        for j in range(na): #asset next period
            for l in range(nl):
              c = (1+r)*agrid[i] + w*lgrid[l] - agrid[j]  
              mat[i,j,l] = CRRA(c)
    return mat   


@njit
def VFI(c0,v0,util,eps=1e-3, diff=1, itermax=300):
    """
    ---------------------------------------------------------------------------
    Solving Bellman equation via Value Function Iteration
    ---------------------------------------------------------------------------
    <input>
    ・ c0 : initial guess of policy function(na-by-nl)
    ・ v0 : initial guess of value function(na-by-nl)
    ・ util: utility matrix(na-by-na-by-nl)
    ・ eps: tolerence
    <output>
    ・ c1 : converged policy function
    ・ v1 : converged value function  
    """
    na, nl = len(agrid), len(lgrid)
    v1 = np.empty_like(v0)
    c1 = np.empty_like(c0)
    num = 0
    while diff>eps: 
        for sp in range(nl): #state this period
            E = np.sum(P[sp,:]*v0, axis=1)
            for i in range(na):
                aj = np.argmax(util[i,:,sp] +beta*E) 
                v1[i,sp] = util[i,aj,sp] + beta*E[aj]
                c1[i,sp] = agrid[aj]
        diff = np.max(np.abs(v1-v0))
        v0 = np.copy(v1)   
        num += 1
        #print(f"number of iteration is {num} and the error is {diff:.7f}")
    return v1, c1


def transition(c1):
    """
    ---------------------------------------------------------------------------
    Construct Transition Matrix (sparse matrix)
    ---------------------------------------------------------------------------
    <input>
    ・ c1: policy function derived from VFI
    <output>
    ・ A: transition matrix, which is sparse matrix
    """
    na, nl = len(agrid), len(lgrid)
    m = na*nl
    A = np.zeros((m,m)) #transition matrix
    
    for i in range(na):
        for k in range(nl):
            j = int(np.where(agrid ==c1[i,k])[0])
            for l in range(nl):
                m = na*k + i      # index for this period
                n = na*l + j      # index for next period(have a prob for both states)
                A[m, n] = P[k,l]
    A = csr_matrix(A) #converting A to sparse matrix to make computation more efficient
    return A


def stationary_distribution(A):
    """
    --------------------------------------------------------------------------
    Returns stationary distribution of asset and state
    --------------------------------------------------------------------------
    <input>
    ・ A: transition matrix(na-by-nl matrix)
    <output>
    ・ mu: stationary distribution
    """
    eig_value,eig_vector = linalg.eigs(A.T,sigma=0.99999)
    minimizer = np.argmin(np.abs(eig_value-1))    
    unit_eig_vector = eig_vector[:,minimizer]
    mu = unit_eig_vector / sum(unit_eig_vector)
    
    return mu.real


def convert_stationary_distribution(mu):
    """
    ---------------------------------------------------------------------------
    Reshape stationary distribution vector to na-by-ns matrix
    ---------------------------------------------------------------------------
    <input>
    ・ mu: stationary distribution(na*nl-by-1 vector)
    ・ model: Aiyagari class
    """
    stationary_mat = np.empty((na, nl))
    for i in range(nl):
        stationary_mat[:,i] = mu[i*na:(i+1)*na]

    return stationary_mat

@njit
def aggregate_N(stationary_mat):
    """
    ---------------------------------------------------------------------------
    Aggregating labor inputs and derive aggregate labor in economy
    <input>
    ・ stationary distribution 
    <output>
    ・ aggregate labor inputs
    ---------------------------------------------------------------------------
    """
    distribution = np.sum(stationary_mat,axis=0)
    N = np.dot(lgrid,distribution)
    return N

@njit    
def market_clearing(mu,stationary_mat):
    """
    --------------------------------------------------------------------------
    Function that checks whther asset market clears, 
    that is the weighted average of the asset holding is zero
    in the stationary distribution
    --------------------------------------------------------------------------
    <input>
    ・ mu: stationary distribution(na*ns-by-1 vector)
    ・ model: Huggett class
    """
    distribution = np.sum(stationary_mat,axis=1) 
    z = np.dot(agrid,distribution)
    
    return z


def findr(c0,v0,r):
    """
    ---------------------------------------------------------------------------
    Finding Equilibrium r that clears market
    ---------------------------------------------------------------------------
    <output>
    ・ z-K: excess supply
    """
    N_stationary_dist = stationary_distribution(P) # stationary distribution of labor
    N = np.dot(lgrid,N_stationary_dist) # total labor supply(labor market clears)
    K = (alpha/(r+delta))**(1/(1-alpha)) * N  # total capital demand
    w = (1-alpha)*(alpha/(r+delta))**(alpha/(1-alpha)) # competitive wage
    
    mat = np.empty((na, na, nl))
    u = utility(mat=mat,r=r,w=w)
    v1, c1 = VFI(c0=c0, v0=v0, util=u)
    A = transition(c1)
    mu = stationary_distribution(A)
    stationary_mat = convert_stationary_distribution(mu)
    z = market_clearing(mu,stationary_mat)
    print([r,w,z-K])
    return z-K

def findr2(c0,v0,r):
    """
    ---------------------------------------------------------------------------
    Finding Equilibrium r that clears market
    ---------------------------------------------------------------------------
    <output>
    ・ K: capital demand
    ・ z: capital supply
    """
    N_stationary_dist = stationary_distribution(P) # stationary distribution of labor
    N = np.dot(lgrid,N_stationary_dist) # total labor supply(labor market clears)
    K = (alpha/(r+delta))**(1/(1-alpha)) * N  # total capital demand
    w = (1-alpha)*(alpha/(r+delta))**(alpha/(1-alpha)) # competitive wage
    
    mat = np.empty((na, na, nl))
    u = utility(mat=mat,r=r,w=w)
    v1, c1 = VFI(c0=c0, v0=v0, util=u)
    A = transition(c1)
    mu = stationary_distribution(A)
    stationary_mat = convert_stationary_distribution(mu)
    z = market_clearing(mu,stationary_mat)
    return K, z




start = time.time()
# finding the market clearing r
c0 = np.zeros((na,nl))
v0 = np.zeros((na,nl))
print("[r,w,z-K]")


findr = partial(findr,c0,v0)
r = bisect(findr, a=0, b=0.05, xtol=1e-4) 
w = (1-alpha)*(alpha/(r+delta))**(alpha/(1-alpha))


# solving the problem with optimal r
mat = np.empty((na, na, nl))
u = utility(mat=mat,w=w,r=r)
v1, c1 = VFI(c0=c0, v0=v0, util=u)
A = transition(c1)
mu = stationary_distribution(A)
stationary_mat = convert_stationary_distribution(mu)
z = market_clearing(mu,stationary_mat)
distribution = np.sum(stationary_mat,axis=1)

end = time.time() - start
print(end)


# value function
#labels = ["$l_{low}$","$l_{mid}$","$l_{high}$"]
fig, ax = plt.subplots()
for l in range(nl):
    ax.plot(agrid, c1[:,l], label= f"$l_{l}$")
ax.plot(agrid,agrid, ls="--")
ax.set(xlabel="Credit level =$a$", ylabel="Next Period's Credit Level=$a'$", title="Policy function")
ax.grid(linestyle="--")
ax.legend()
plt.show()


# stationary distribution(marginal)
fig, ax = plt.subplots()
ax.bar(agrid, distribution, width=1)
#ax.plot(agrid, distribution)
ax.set(xlabel="Credit level=$a$", ylabel="marginal asset distribution", title="Stationary distribution(marginal)")
plt.show()

#stationary distribution(cumlative)
fig, ax = plt.subplots()
cumdist = np.cumsum(distribution)
ax.plot(agrid,cumdist)
ax.set(xlabel="Credit Level = $a$", title="Stationary distribution(cumulative)")
plt.show()


r = np.linspace(0.0001, 0.1, 20)
demand = np.empty(len(r))
supply = np.empty(len(r))

for i in tqdm(range(len(r))):
    demand[i], supply[i] = findr2(c0, v0, r[i])
    
fig, ax = plt.subplots()
ax.plot(demand, r, label="demand for capital")
ax.plot(supply, r, label="supply of capital")
ax.set(xlabel="capital", ylabel="interest rate", xlim=(0,12))
ax.legend()
ax.grid(ls="--")
plt.show()



