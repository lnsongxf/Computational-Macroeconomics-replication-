# -*- coding: utf-8 -*-
"""
Replication of Huggett(1993) with time iteration

@author: Taiki Ono
"""

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import bisect
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, linalg
import time

# 1. Parameter Setup 

beta = 0.99322 # discount factor
sigma = 1.5    # relative risk aversion (parameter of CRRA utility function)
na = 1000      # number of grid points of asset
ns = 2         # number of states
eh = 1         # endowment if the state is high
el = 0.1       # endowment if the state is low
ph = 0.925     # transition probability from state high to state high(P_HH)
pl = 0.500     # transition probability from state low to state low(P_LL)
abar = -5.3    # borrowing constaraint
amax = 12      # maximum value of agrid

egrid = np.array([eh, el])
agrid = np.linspace(abar,amax,na)
P = np.array(([[ph, 1-ph], [1-pl, pl]]))


# 2. build functions for computation 
@jit("f8(f8)")
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
        return cons**(1-sigma) / (1-sigma)
    else:
        return -100000

@njit
def utility(mat,q):
    """
    ---------------------------------------------------------------------------
    Returns utility matrix given asset of two periods and states 
    ---------------------------------------------------------------------------
    <input>
    ・ mat: empty matrix(na-by-na-by-ns matrix)
    ・ q: asset price
    <output>
    ・ utility matrix (na-by-na-by-ns matrix)
    where ・ na: number of asset grids(i: this period, j: next period)
          ・ ns: number of states 
    """
    na, ns = len(agrid), len(egrid)
    
    for i in range(na): #asset this period
        for j in range(na): #asset next period
            for s in range(ns):
              c = agrid[i] + egrid[s] -q*agrid[j]  
              mat[i,j,s] = CRRA(c)
    return mat   

@njit
def VFI(c0,v0,util,eps=1e-3, diff=1, itermax=300):
    """
    ---------------------------------------------------------------------------
    Solving Bellman equation via Value Function Iteration
    ---------------------------------------------------------------------------
    <input>
    ・ model: Huggett class
    ・ c0 : initial guess of policy function(na-by-ns)
    ・ v0 : initial guess of value function(na-by-ns)
    ・ util: utility matrix(na-by-na-by-ns)
    ・ eps: tolerence
    <output>
    ・ c1 : converged policy function
    ・ v1 : converged value function  
    """
    
    na, ns = len(agrid), len(egrid)
    v1 = np.empty_like(v0)
    c1 = np.empty_like(c0)
    num = 0
    
    while diff>eps : 
        for s in range(ns):
            E = P[s,0]*v0[:,0] + P[s,1]*v0[:,1] #expected value 
            for i in range(na):
                aj = np.argmax(util[i,:,s] +beta*E) 
                v1[i,s] = util[i,aj,s] + beta*E[aj]
                c1[i,s] = agrid[aj]
        diff = np.max(np.abs(v1-v0))
        v0 = np.copy(v1)   
        num += 1
        #print(f"number of iteration is {num} and the error is {diff:.7f}")
    return v1, c1


def transition(c1):
    """
    ---------------------------------------------------------------------------
    Construct Transition Matrix
    ---------------------------------------------------------------------------
    <input>
    ・ model: Huggett class
    ・ c1: policy function derived from VFI
    """
    na, ns = len(agrid), len(egrid)
    A = np.zeros((na*ns, na*ns)) #transition matrix
    
    for i in range(na):
        for k in range(ns):
            j = int(np.where(agrid ==c1[i,k])[0])
            for l in range(ns):
                m = na*k + i      # index for this period
                n = na*l + j      # index for next period(have a prob for both states)
    
                A[m, n] = P[k,l]
    A = csr_matrix(A) #converting A to sparse matrix to make computation efficient
    return A

def stationary_distribution(A):
    """
    --------------------------------------------------------------------------
    Returns stationary distribution of asset and state
    --------------------------------------------------------------------------
    <input>
    ・ A: transition matrix(na-by-na matrix)
    """
    eig_value,eig_vector = linalg.eigs(A.T, sigma=1)
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
    ・ mu: stationary distribution(na*ns-by-1 vector)
    ・ model: Huggett class
    """
    stationary_mat = np.empty((na, ns))
    stationary_mat[:,0] = mu[0:na]
    stationary_mat[:,1] = mu[na:2*na]
    
    return stationary_mat


def market_clearing(mu):
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

    z = np.sum(agrid*mu[0:na]) + np.sum(agrid*mu[na:2*na])
    
    return z

#@jit(nopython=True)
def findq(c0,v0,q):
    """
    ---------------------------------------------------------------------------
    Finding Equilibrium q that clears market
    ---------------------------------------------------------------------------
    """
    na, ns = len(agrid), len(egrid)
    mat = np.empty((na, na, ns))
    u = utility(mat=mat,q=q)
    v1, c1 = VFI(c0=c0, v0=v0, util=u)
    A = transition(c1)
    mu = stationary_distribution(A)
    z = market_clearing(mu)
    print([q, z])
    return z

# 3. Implementation(finding equilibrium price)   
start = time.time()

c0 = np.zeros((na,ns))
v0 = np.zeros((na,ns))  

findq = partial(findq,c0,v0)
q = bisect(findq, a=0.99, b=1, xtol=1e-4)

end = time.time() -start
print(f"running time is {end}") 

# 4. plotting policy function and stationary distribution given optimal q
mat = np.empty((na, na, ns))
u = utility(mat=mat,q=q)
v1, c1 = VFI(c0=c0, v0=v0, util=u)
A = transition(c1)
mu = stationary_distribution(A)
stationary_mat = convert_stationary_distribution(mu)
z = market_clearing(mu)


# policy function
labels = ["$e_h$", "$e_l$"]
fig, ax = plt.subplots()
for s in range(ns):
    ax.plot(agrid, c1[:,s], label=labels[s])
ax.plot(agrid,agrid, "--", c="k")
ax.axhline(y=0,ls="--", c="k")
ax.axvline(x=0,ls="--", c="k")
ax.set(xlabel="Credit level =$a$", ylabel="Next Period's Credit Level", title="Policy function",
       xlim=(-3,12), ylim=(-3,12))
ax.legend()
plt.show()

# fig, ax = plt.subplots()
# for s in range(ns):
#     cons = egrid[s] + agrid - q*c1[:,s]
#     ax.plot(agrid, cons, label=labels[s])
# ax.axvline(x=0, ls="--", c="k")
# ax.set(xlabel="Credit level =$a$", ylabel="Consumption policy", xlim=(-3,12), ylim=(0,2))
# ax.legend()
# plt.show()

# stationary distribution(marginal)
fig, ax = plt.subplots()
#ax.plot(agrid, stationary_mat[:,0]+stationary_mat[:,1])
ax.bar(agrid, stationary_mat[:,0]+stationary_mat[:,1],width=1)
ax.set(xlabel="Credit level=$a$", ylabel="marginal asset distribution", title="Stationary distribution(marginal)")
plt.show()

#stationary distribution(cumlative)
fig, ax = plt.subplots()
cumdist = np.cumsum(stationary_mat[:,0]+stationary_mat[:,1])
ax.plot(agrid,cumdist)
ax.set(xlabel="Credit Level = $a$", title="Stationary distribution(cumulative)")
plt.show()
    
    
    
    
    
    
    
    
    
    
    