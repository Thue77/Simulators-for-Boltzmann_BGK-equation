import numpy as np
from .one_step import phi_KD,__psi_k
from typing import Callable,Tuple
from numba import njit
from numba import prange
import time
from scipy.stats import wasserstein_distance

'''Standard Monte Carlo method'''


#Function to return a copy of the given numpy array that has been given values 'new' at 'index'
@njit(nogil=True)
def __put_copy(self,arr,index,new):
    out = arr.copy()
    out[index] = new
    return out

#The KDMC method with the use of a step function
@njit(nogil=True)
def KDMC(dt,N,Q,t0,T,mu,sigma,M,R,SC,Nested =False,dR=None,boundary=None,x0=None,v0=None):
    '''
    dt: step size
    x0: initial positions
    v0: initial velocities
    e: exponential number to generate first collision time
    tau: first collision time for each particle
    t0: starting time. Same for all particles
    T: end time. Same for all particles
    mu: mean of post-collisional distribution
    sigma: standard deviation of post-collisional distribution
    M: post-collisional distribution
    R: collision rate
    SC: method for obtaining the next collision times
    '''
    t = np.ones(N)*t0
    I = np.ones(N)==1;tau = np.ones(N)*T
    if x0 is None and v0 is None:
        x,v,_ = Q(N)
    else:
        x = x0.copy();v=v0.copy()
    while True:
        e = np.random.exponential(1,size=np.sum(I,dtype=np.int64))
        tau[I] = SC(x[I],v[I],e)
        I = (t+tau)<=T #Indicate active paths
        if np.sum(I)==0:
            break
        # n = len(index)
        # if n==0:
        #     break
        xi = np.random.normal(0,1,size=np.sum(I))
        x[I],v[I],t[I],_ = phi_KD(dt,x[I],v[I],t[I],tau[I],xi,mu,sigma,M,R,dR=dR,boundary=boundary)
        #Update first collision time for next step
        # e = np.random.exponential(1,size=np.sum(I))
        # tau[I] = SC(x[I],v[I],e)
        # t[index] = t_temp; x[index] = x_temp; v[index] = v_temp
        # I = (t+tau)<T
        # index = np.argwhere(I).flatten()
    I = t<T
    if np.sum(I)>0: #Move the rest of the particles kinetically to the end
        index = np.argwhere(I).flatten()
        x[index] = x[index] + v[index]*(T-t[index])
        if boundary is not None: x = boundary(x)
        t[index] = T
    return x

@njit(nogil=True)
def Kinetic(N,Q,t0,T,mu,sigma,M,R,SC,boundary=None,x0=None,v0=None):
    if x0 is None and v0 is None:
        x,v,_ = Q(N)
    else:
        x = x0.copy();v=v0.copy()
    t = t0*np.ones(N)
    I = np.ones(N)==1;tau = np.ones(N)*T
    while True:
        e = np.random.exponential(1,size=np.sum(I,dtype=np.int64))
        tau[I] = SC(x[I],v[I],e)
        I = (t+tau)<=T #Indicate active paths
        if np.sum(I)==0:
            break
        x[I],t[I] = __psi_k(tau[I],x[I],v[I],t[I])
        if boundary is not None: x = boundary(x)
        v[I],_ = M(x[I])
    I = (T-t)>0
    if np.sum(I)>0:
        x[I] = x[I] + v[I]*(T-t[I])
        if boundary is not None: x = boundary(x)
    return x

@njit(nogil=True,parallel=True)
def mc1_par(dt,N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0=None,v0=None):
    cores = 8
    n = round(N/cores)
    x_KD = np.empty((cores,n))
    for i in prange(cores):
        if x0 is None and v0 is None:
            x_KD[i,:] = KDMC(dt,n,Q,t0,T,mu,sigma,M,R,SC,dR=dR,boundary=boundary)
        else:
            x_KD[i,:] = KDMC(dt,n,Q,t0,T,mu,sigma,M,R,SC,dR=dR,boundary=boundary,x0=x0[i*n:(i+1)*n],v0=v0[i*n:(i+1)*n])
    return x_KD.flatten()

@njit(nogil=True,parallel=True)
def mc2_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0=None,v0=None):
    cores = 8
    n = round(N/cores)
    x_std = np.empty((cores,n))
    for i in prange(cores):
        if x0 is None and v0 is None:
            x_std[i,:] = Kinetic(n,Q,t0,T,mu,sigma,M,R,SC,boundary=boundary)
        else:
            x_std[i,:] = Kinetic(n,Q,t0,T,mu,sigma,M,R,SC,boundary=boundary,x0=x0[i*n:(i+1)*n],v0=v0[i*n:(i+1)*n])
    return x_std.flatten()

def mc_density_test(dt_list,Q,t0,T,N,mu,sigma,M,R,SC,dR=None,boundary=None,x_std=None,x0=None,v0=None):
    '''Returns a wasserstein distance and the associated standard deviation'''
    W_out = np.zeros(dt_list.size); err = np.zeros(dt_list.size); cost = np.zeros(dt_list.size)
    if x0 is None and v0 is None:
        x0,v0,_ = Q(N)
    if x_std is None: x_std = mc2_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0=x0,v0=v0)
    for j,dt in enumerate(dt_list):
        W = np.zeros(20)
        print(dt)
        start = time.perf_counter()
        for i in range(20):
            x_KD = mc1_par(dt,N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0=x0,v0=v0)
            W[i] = wasserstein_distance(x_KD,x_std)
        cost[j] = time.perf_counter()-start
        W_out[j] = np.mean(W); err[j] = np.std(W)
    return W_out,err,cost
