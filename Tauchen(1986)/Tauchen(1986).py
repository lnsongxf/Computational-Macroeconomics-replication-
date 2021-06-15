# -*- coding: utf-8 -*-
"""
Approximating AR(1) process by Tauchen(1986) method

@author: Taiki Ono
"""
import numpy as np
import scipy.stats as stats
from quantecon import MarkovChain
import matplotlib.pyplot as plt

def tauchen_approximation(rho,sigma_u, b=0,  m=3, n=7):
    """
    ----------------------------------------------------------------------
    Approximating AR(1) process to finite Markov chain by Tauchen method
    y_{t+1} = b + rho*y_{t} + u{t+1}
    ----------------------------------------------------------------------
    <input>
    ・ b: constant term in AR(1) process
    ・ rho: the autocorrelation coefficient
    ・ sigma_u: standard deviation of random variable
    ・ m: multiplier for the interval of nodes
    ・ n: number of nodes
    <output>
    ・ z: uniform nodes
    ・ pi: Markov chain(n-by-n matrix)
    """
    # unconditional standard deviation of y
    sigma_y = np.sqrt( sigma_u**2 / (1-rho**2) )
    # construct uniform nodes
    z = np.linspace(-m*sigma_y, m*sigma_y, n) 
    
    #interval of nodes and mid point of the nodes
    interval_z = z[1] - z[0]
    interval_m = interval_z / 2
    
    # midpoint of z_{i} and z_{i+1}
    m = np.empty(n-1) 
    for i in range(n-1):
        m[i] = z[i] + interval_m
        
    # construct markov chain by Tauchen method
    pi = np.empty((n,n)) 
    
    for j in range(n):
        # for first and last columns
        pi[j, 0] = stats.norm.cdf( (m[0]-b-rho*z[j]) / sigma_u )
        pi[j, -1] = 1 -  stats.norm.cdf( (m[-1]-b-rho*z[j]) / sigma_u )
        for k in np.arange(1,n-1):
        # for 2~n-1 columns
            range_max = m[k]-b-rho*z[j]
            range_min = m[k-1]-b-rho*z[j]
            pi[j,k] = stats.norm.cdf(range_max / sigma_u) - stats.norm.cdf(range_min / sigma_u)
    
    return z, pi


# simulate markov chain and compare it with AR(1) process
def AR1_simurate(rho,sigma_u,b=0,y0=0, n=250):
    """
    --------------------------------------------------------------------------
    simulate AR(1) process
    y_{t+1} = b + rho*y_{t} + e_{t+1}
    where e_{t+1} ~ N(0,sigma_u**2)
    --------------------------------------------------------------------------
    <input>
    ・ b: constant term in AR(1) process
    ・ rho: the autocorrelation coefficient
    ・ sigma_u: standard deviation of random variable
    ・ y0: initial value of the simulation
    ・ n: simulation period
    <output>
    ・ simulated values
    """
    simulated = np.empty(n)
    simulated[0] = y0
    for t in range(n-1):
        simulated[t+1] = b + rho*simulated[t] +stats.norm(scale=sigma_u).rvs(1)
    
    return simulated

rho = 0.95
sigma_u = 0.10
n = 250

z, pi = tauchen_approximation(rho,sigma_u,n=35)
mc = MarkovChain(pi,state_values=z)

X = mc.simulate(250, init=0)
AR = AR1_simurate(rho, sigma_u,n=n)

fig, ax = plt.subplots()
ax.plot(X, label="Markov Chain")
ax.plot(AR, label="AR1")
ax.set(xlabel="period",ylabel="$y_{t}$", title="Accuracy of Tauchen method")
ax.legend()
plt.show()