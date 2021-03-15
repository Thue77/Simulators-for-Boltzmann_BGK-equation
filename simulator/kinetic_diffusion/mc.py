import numpy as np
from .one_step import phi_KD,__psi_k
from typing import Callable,Tuple
from numba import njit

'''Standard Monte Carlo method'''


#Function to return a copy of the given numpy array that has been given values 'new' at 'index'
@njit(nogil=True)
def __put_copy(self,arr,index,new):
    out = arr.copy()
    out[index] = new
    return out

#The KDMC method with the use of a step function
@njit(nogil=True)
def KDMC(dt,x0,v0,e,tau,t0,T,mu:Callable[[np.ndarray],np.ndarray],sigma:Callable[[np.ndarray],np.ndarray],M:Callable[[np.ndarray,int],np.ndarray],R:Callable[[np.ndarray],np.ndarray],SC:Callable[[int],np.ndarray],Nested =False,dR=None):
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
    n = x0.size
    r = x0.size
    n_old = n
    t = np.ones(n)*t0
    I = (t+tau)<T
    index = np.argwhere(I).flatten()
    x = x0.copy(); v = v0.copy()
    count = 0
    while True:
        n = len(index)
        if n==0:
            break
        xi = np.random.normal(0,1,size=len(index))
        x_temp,v_temp,t_temp,_ = phi_KD(dt,x[index],v[index],t[index],tau[index],xi,mu,sigma,M,R,dR=dR)
        #Update first collision time for next step
        e = np.random.exponential(1,size=len(index))
        tau[index] = SC(x[index],v[index],e)
        t[index] = t_temp; x[index] = x_temp; v[index] = v_temp
        I = (t+tau)<T
        index = np.argwhere(I).flatten()
        count += 1
    I = t<T
    if np.sum(I)>0: #Move the rest of the particles kinetically to the end
        index = np.argwhere(I).flatten()
        x[index] = x[index] + v[index]*(T-t[index])
        t[index] = T
    return x

# @njit(nogil=True)
def Kinetic(N,Q,t0,T,mu:Callable[[np.ndarray],np.ndarray],sigma:Callable[[np.ndarray],np.ndarray],M:Callable[[np.ndarray,int],np.ndarray],R:Callable[[np.ndarray],np.ndarray],SC:Callable[[int],np.ndarray]):
    x,v,_ = Q(N)
    t = t0*np.ones(N)
    I = np.ones(N).astype(np.bool);tau = np.ones(N)*T
    while True:
        e = np.random.exponential(1,size=np.sum(I))
        tau[I] = SC(x[I],v[I],e)
        I = (t+tau)<=T #Indicate active paths
        if np.sum(I)==0:
            break
        x[I],t[I] = __psi_k(tau[I],x[I],v[I],t[I])
        v[I] = mu(x[I])+sigma(x[I])*M(x[I])
    I = (T-t)>0
    if np.sum(I)>0:
        x[I] = x[I] + v[I]*(T-t[I])
    return x
