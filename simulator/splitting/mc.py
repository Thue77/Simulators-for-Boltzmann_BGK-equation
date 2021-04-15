from .one_step import phi_APS,phi_standard
import numpy as np
from numba import njit


@njit(nogil=True)
def mc(dt,t0,T,N,eps,Q,M,r=None,boundary = None,sigma=None):
    t = t0
    x,v,_ = Q(N)
    while t<T:
        active = t<T
        z = np.random.normal(0,1,size=N); u = np.random.uniform(0,1,size=N)
        if r is None:
            x,v,_ = phi_APS(x,v,dt,eps,z,u,M,boundary=boundary)
        else:
            x,v,_ = phi_APS(x,v,dt,eps,z,u,M,r=r(x),boundary=boundary)
        t += dt
    return x

@njit(nogil=True)
def mc_standard(dt,t0,T,N,eps,Q,M,boundary,r):
    t = t0
    x,v,_ = Q(N)
    while t<T:
        active = t<T
        # z = np.random.normal(0,1,size=N); u = np.random.uniform(0,1,size=N)
        x,v = phi_standard(x,v,dt,eps,M,boundary=boundary,r=r)
        t += dt
    return x
