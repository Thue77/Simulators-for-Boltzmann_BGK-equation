from .one_step import phi_APS
import numpy as np


def mc(dt,t0,T,N,eps,Q):
    t = t0
    x,v,_ = Q(N)
    while t<T:
        active = t<T
        z = np.random.normal(size=N); u = np.random.uniform(size=N)
        x,v = phi_APS(x,v,dt,eps,z,u)
        t += dt
    return x
