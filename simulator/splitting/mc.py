from .one_step import phi_APS
import numpy as np


def mc(dt,t0,T,N,eps,Q,B,boundary = None,sigma=None):
    t = t0
    x,_,v = Q(N)
    if sigma is not None:
        v = v*eps/(eps**2+dt)*sigma(x)*eps
    while t<T:
        active = t<T
        z = np.random.normal(size=N); u = np.random.uniform(size=N)
        x,v,_ = phi_APS(x,v,dt,eps,z,u,B,boundary=boundary,sigma=sigma)
        t += dt
    return x
