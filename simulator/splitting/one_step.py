import numpy as np
from numba import njit,jit_module,prange



def psi_t(x,v,dt,eps,z,r):
    # print(f'v in transport term: {v}')
    # print(dt)
    # D = (v*eps/(eps**2+dt*r))**2*dt/(eps**2+dt*r)
    # return x + v*eps/(eps**2+dt*r)*dt +   np.sqrt(2*dt)*np.sqrt(D)*z#np.sqrt(2*(v*eps/(eps**2+dt*r))**2*dt**2/(eps**2+dt*r))*z
    '''The characteristic velocity is not included in the diffusion coefficient
    and must be divided out'''
    D = v**2*dt/(eps**2+dt*r(x))#(v*(eps**2+dt*r)/eps)**2*dt/(eps**2+dt*r)
    A = eps/(eps**2+dt*r(x))*v
    return x + A*dt + np.sqrt(2*dt*D)*z#v*dt + np.sqrt(2*dt*D)*z

def psi_c(x,v,dt,eps,u,B,r,v_bar = None):
    '''
    u: vector of uniform numbers in [0,1] to sample from M with appropriate probability
    '''
    p = (u>=eps**2/(eps**2+dt*r(x))) #1 if collision occurs
    # print(f'dt: {dt}, prob: {1-eps**2/(eps**2+dt*r)}')
    if v_bar is None:
        _,v_bar = B(x)

    v = (1-p)*v + p*v_bar#v_char(dt,eps,v_tilde)*p*v_bar
    # print(f'v_bar: {v_bar}, v: {v}')
    # if dt ==1: print(1-p)
    return v,v_bar

# def v_char(dt,eps,v_tilde=1,r=1):
#     '''
#     v_tilde is the standard deviation of the post-collisional distribution
#     '''
#     return eps/(eps**2+dt*r)*v_tilde

def phi_APS(x,v,dt,eps,z,u,B,r,v_next=None,boundary=None):
    # print(f'x= {x} \n v= {np.var(v)}')
    x = psi_t(x,v,dt,eps,z,r)
    # if sigma is not None:
    #     v_tilde = sigma(x)*eps
    # else:
    #     v_tilde=1
    if boundary is not None:
        x = boundary(x)
    v,v_bar = psi_c(x,v,dt,eps,u,B,r,v_next)
    return x,v,v_bar


def phi_standard(x,v,dt,eps,M,r,boundary):
    #transport step
    x = boundary(x+v/eps*dt)
    #collision step
    u = np.random.uniform(0,1,size=len(x))
    p = u>np.exp(-r(x)/eps**2*dt)
    _,v_bar = M(x)
    v = (1-p)*v + p*v_bar
    return x,v


# jit_module(nopython=True,nogil=True)



def phi_SS(x,v,dt,eps):
    # x = v/eps*dt
    u = np.random.uniform(size=len(x))
    x = x + np.sqrt(2*dt*v**2)*np.random.normal(size=len(x))
    v = eps/(eps**2+dt)*np.random.uniform(-1,1,size=len(x))*(u>=eps**2/(eps**2+dt)) + v*(u<eps**2/(eps**2+dt))
    return x,v
