import numpy as np




def psi_t(x,v,dt,eps,z,r=1):
    return x+ v*eps/(eps**2+dt*r)*dt + np.sqrt(2*v**2*dt**2/(eps**2+dt*r))*z


def psi_c(x,v,dt,eps,M,u,r=1):
    '''
    u: vector of uniform numbers to sample from M with appropriate probability
    '''
    p = (u<=dt*r/(eps**2+dt*r))
    v = (1-p)*v + p*M(x)
    return v

def phi_APS(x,v,dt,eps,z,M,u,r=1):
    x = psi_t(x,v,dt,eps,z,r=1)
    v = psi_c(x,v,dt,eps,M,u,r=1)
    return x,v
