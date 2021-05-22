import numpy as np
import sys
from numba import njit

#%%
n = 100000
x=np.zeros(n); v=np.ones(n)#np.random.normal(size=n);
def M(x):
    if True:
        return np.random.normal(size=x.size)
    else:
        U = np.random.uniform(0,1,size=len(x))
        return  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)


r=lambda x:np.zeros(x.size)
eps = 0.01
#%%
def psi_t(x,v,dt,eps,z,r,diff=False,v_ms=1):
    '''v_ms: <nu^2>'''
    if diff:
        D = v_ms*dt/(eps**2+dt*r(x))#
    else:
        D = v**2*dt/(eps**2+dt*r(x))#
    A = eps/(eps**2+dt*r(x))*v
    return x + A*dt + np.sqrt(2*dt*D)*z

def psi_c(x,v,dt,eps,u,B,r,v_next = None):
    '''
    u: vector of uniform numbers in [0,1] to sample from M with appropriate probability
    '''
    p = (u>=eps**2/(eps**2+dt*r(x))) #1 if collision occurs
    if v_next is None:
        v_next = B(x)

    v = (1-p)*v + p*v_next#v_bar
    return v

def phi_APS(x,v,dt,eps,z,u,B,r,v_next=None,diff=False,v_ms=1):
    x = psi_t(x,v,dt,eps,z,r,diff=diff,v_ms=v_ms)
    v = psi_c(x,v,dt,eps,u,B,r,v_next)
    return x,v


def correlated(dt_f,M_t,t,T,eps,x0,v0,B,r,diff=False,v_ms=1):
    '''
    M_t: defined s.t. dt_c=M_t dt_f
    t: starting time
    eps: diffusive parameter
    N: number of paths
    Q: Initial distribution
    M: velocity distribution
    '''
    dt_c = dt_f*M_t
    first_level = np.abs(T-dt_c)<1e-7
    # if first_level:
    #     print(f'dt_f: {dt_f}, dt_c: {dt_c}, M_t: {M_t}')
    N = x0.size
    x_f,v_f = x0.copy(),v0.copy()
    x_c = x_f.copy()
    v_bar_c = v_f.copy()
    v_c = v_f.copy()
    while t<T:
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))
        for m in range(M_t):
            C = (U.T>=eps**2/(eps**2+dt_f*r(x_f))).T #Indicates if collisions happen
            x_f,v_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r,diff=diff,v_ms=v_ms)
            v_bar_c[C[:,m]] = v_f[C[:,m]]
        z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c,diff=diff,v_ms=v_ms)
        t += dt_c
    return x_f,x_c

def max_np(A,axis=1):
    '''Returns array with maximum values along given axis. numba replace for np.max'''
    shape = np.shape(A)
    n = shape[1-axis]; m=shape[axis]
    out = np.zeros(n)
    for j in range(m):
        out = np.maximum(A[:,j],out)
    return out

#%%
dt_f = 1/2**3; M_t=2
x_f,x_c = correlated(dt_f,M_t,0,1,eps,x,v,M,r,diff=True,v_ms=1)

#%%
np.var(x_f-x_c)

np.log(1706.6746117126527)
#%%

if __name__ == '__main__':

    M_t = 5;
    C = np.array([[False, False,  True, False,  True],[False, False,  True, False,  True]])
    v_bar_all = np.array([[ 0.,          0.,         -0.4226053,   0.,          0.93366715],[ 0.,          0.,         -0.4226053,   0.,          0.93366715]])
    goal = np.array([[0,0,v_bar_all[0,2]*np.sqrt(2/3),0,v_bar_all[0,4]*np.sqrt(1/3)],[0,0,v_bar_all[0,2]*np.sqrt(2/3),0,v_bar_all[0,4]*np.sqrt(1/3)]])
    np.sum(goal,axis=1)
    print(f'result: {cor_rv(M_t,C,v_bar_all)}')
