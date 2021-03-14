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
def select_levels(L,Q,t0,T,mu,sigma,M,R,SC,R_anti=None,dR=None,N=100,tau=None):
    '''
    V: array of variances. Length L+1
    V_d: array of variances of bias. Length L
    output:
    let l be in levels[1:]. Then dt^f = 1/2**l.
    for l=levels[0], dt = 1/2**l
    '''
    Q_est,Q_d,V,V_d,C,C_d = warm_up(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N)
    l = 1
    test = V_d > V[1:]
    if np.sum(test)>1:
        l = np.argwhere(test).flatten()[-1]+2 #Last index where variance of bias is larger than V
    levels = [l-1,l]
    V_min = V_d[max(l-2,0)]
    for j in range(l,len(V_d)):
        if V_d[j]<V_min/2:
            levels += [j]
            V_min = V_d[j]
    L_set = np.array(levels)
    '''Set up output variables based on level selection and set values of non adjecant levels to zero'''
    Q_out = np.empty(len(levels)) #List of ML estimates for each level
    V_out = np.empty(len(levels)) #Variances of estimates on each level
    C_out = np.empty(len(levels)) #Cost of estimates on each level
    Q_out[0] = Q_est[levels[0]]; V_out[0] = V[levels[0]]; C_out[0] = C[levels[0]] #Values for first level
    '''Note that len(Q_l_l1)=L and len(Q_l)=L+1. So the first value in Q_l_l1 is Q_{1,0}.
    Hence, if level 2 and 3 are included we want Q_{3,2}, which is at Q_l_l1[2]
    '''
    '''First determine jumps in levels'''
    jumps = np.where(diff_np(L_set)>1)[0] #index for jumps in terms of correlated results
    Q_temp = Q_d[L_set[1:]-1]; V_temp = V_d[L_set[1:]-1]; C_temp = C_d[L_set[1:]-1]
    '''No results are available for non-adjecant levels. So they are set to 0'''
    Q_temp[jumps] = 0; V_temp[jumps] = 0; C_temp[jumps] = 0;
    '''Insert in output variables'''
    Q_out[1:] = Q_temp; V_out[1:] = V_temp; C_out[1:] = C_temp; #Values for other levels
    '''Set number of paths for each level'''
    N_out = N*np.where(C_out > 0)[0]
    return levels,N_out,Q_out,V_out,C_out

@njit(nogil=True)
def diff_np(a):
    return a[1:]-a[:-1]

def ml(e2,Q,t0,T,mu,sigma,M,R,SC,R_anti=None,dR=None,N=100,tau=None,L=14,N_warm = 100):
    '''First do warm-up and select levels with L being the maximum level'''
    levels,N,X,V,C = select_levels(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N_warm,tau)
