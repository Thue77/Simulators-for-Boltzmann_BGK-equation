from .one_step import phi_APS
import numpy as np


def mc(dt,t0,T,N,eps,Q,B,boundary = None):
    t = t0
    x,v,_ = Q(N)
    while t<T:
        active = t<T
        z = np.random.normal(size=N); u = np.random.uniform(size=N)
        x,v,_ = phi_APS(x,v,dt,eps,z,u,B)
        if boundary is not None: x = boundary(x)
        t += dt
    return x
