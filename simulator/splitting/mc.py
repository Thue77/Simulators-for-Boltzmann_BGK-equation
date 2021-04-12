from .one_step import phi_APS
import numpy as np


def mc(dt,t0,T,N,eps,Q,M,r=None,boundary = None,sigma=None):
    t = t0
    x,_,v = Q(N)
    while t<T:
        active = t<T
        z = np.random.normal(size=N); u = np.random.uniform(size=N)
        if r is None:
            x,v,_ = phi_APS(x,v,dt,eps,z,u,M,boundary=boundary)
        else:
            x,v,_ = phi_APS(x,v,dt,eps,z,u,M,r=r(x),boundary=boundary)
        t += dt
    return x
