import numpy as np
from .correlated import correlated
from .mc import KDMC
import time
from numba import njit,jit_module,prange,objmode

@njit(nogil=True,parallel=True)
def warm_up(L,Q,t0,T,mu,sigma,M,R,SC,R_anti=None,dR=None,N=100,tau=None):
    dt_list = 1/2**np.arange(0,L+1)
    Q_l = np.zeros(L+1) #Estimates for all possible first levels
    Q_l_L = np.zeros(L) #Estimates for all adjecant biases
    V_l = np.zeros(L+1)
    V_l_L = np.zeros(L)
    C_l = np.zeros(L+1) ##Cost for each level
    C_l_L = np.zeros(L)
    x0,v0,v_l1_next = Q(N)
    e = np.random.exponential(scale=1,size=N)
    tau = SC(x0,v0,e)
    for l in prange(L+1):
        if l < L:
            with objmode(start1='f8'):
                start1 = time.perf_counter()
            x_f,x_c = correlated(dt_list[l+1],dt_list[l],x0,v0,v_l1_next,t0,T,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR)
            with objmode(end1='f8'):
                end1 = time.perf_counter()
            C_l_L[l] = end1-start1
            x_dif = x_f-x_c
            Q_l_L[l] = np.mean(x_dif)
            V_l_L[l] = np.var(x_dif)
        with objmode(start2='f8'):
            start2 = time.perf_counter()
        x = KDMC(dt_list[l],x0,v0,e,tau,0,T,mu,sigma,M,R,SC)
        with objmode(end2='f8'):
            end2 = time.perf_counter()
        C_l[l] = end2-start2
        Q_l[l] = np.mean(x)
        V_l[l] = np.var(x)
    return Q_l,Q_l_L,V_l,V_l_L,C_l,C_l_L


@njit(nogil=True)
def select_levels(V,V_d):
    '''
    V: array of variances. Length L+1
    V_d: array of variances of bias. Length L
    '''
    l = 1
    test = V_d > V[1:]
    if np.sum(test)>1:
        l = np.argwhere(test).flatten()[-1]+2 #Last index where variance of bias is larger than V
    L = [l-1,l]
    V_min = V_d[max(l-2,0)]
    for j in range(l,len(V_d)):
        if V_d[j]<V_min/2:
            L += [j]
            V_min = V_d[j]
    return L
