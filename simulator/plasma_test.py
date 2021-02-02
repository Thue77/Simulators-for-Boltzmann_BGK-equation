import numpy as np
from numba import njit
from globals import initialize
from typing import Callable,Tuple

initialize()

@njit
def M(x):
    return np.random.normal(0,1,size=x.size)


#Inintial distribution of position and velocity
@njit
def Q(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # np.random.seed(4208)
    U = np.random.uniform(0,1,size=N)
    I = np.random.uniform(0,1,size=N)>0.5
    index = np.argwhere(I).flatten()
    index_not = np.argwhere(np.logical_not(I)).flatten()
    x = np.ones(N)
    v = np.zeros(N)
    v_norm = np.append(np.random.normal(0,1,size = len(index)),np.random.normal(0,1,size = N-len(index)))
    v[index] = (v_norm[0:len(index)] + 10)/epsilon
    v[index_not] = (v_norm[len(index):]-10)/epsilon
    return x,v,v_norm

#sets the collision rate
@njit
def R(x):
    if type == 'default':
        return 1/(epsilon**2)
    elif type == 'B1':
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)


#Simpler version of SampleCollision
@njit
def SC(x,v,e):
    dtau = 1/R(x)*e
    return dtau

@njit
def mu(x):
    return 0

@njit
def sigma(x):
    return 1/epsilon
