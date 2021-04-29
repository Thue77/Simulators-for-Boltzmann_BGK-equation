import numpy as np
from .correlated import correlated
from .mc import mc
from .AddPaths import delta,x_hat,Sfunc
import time
from numba import njit,jit_module,prange,objmode

# @njit(nogil=True)
def select_levels(t0,T,M_t,eps,F,strategy=1,cold_start=True,N=100,boundary=None):
    '''
    strategy: indicates if strategy 1 or 2 is used
    cold_start: indicates if the variance and estimates are calculated for an
    initial number of paths
    '''
    if strategy==1:
        dt_0 = np.minimum(eps**2,0.025)
        levels = [dt_0]
        levels += [dt_0/M_t]
        N_out=np.zeros(2,dtype=np.int64)
        N_diff=np.ones(2,dtype=np.int64)*N
        SS_out = np.zeros(2);C_out = np.zeros(2); E_out=np.zeros(2)
    else:
        levels = [float(T-t0)]
        dt_1 = np.minimum(eps**2,0.025)
        levels += [dt_1]
        levels += [dt_1/M_t]
        N_out=np.zeros(3,dtype=np.int64)
        N_diff=np.ones(3,dtype=np.int64)*N
        SS_out = np.zeros(3);C_out = np.zeros(3); E_out=np.zeros(3)
    return levels,N_out,N_diff,E_out,SS_out,C_out


@njit(nogil=True,parallel=True)
def update_paths(I,E,SS,C,N,N_diff,levels,t0,T,M_t,eps,Q,M,r,F,boundary,strategy=1):
    for j in prange(len(I)):
        i=I[j]
        # print(i)
        dt_f = levels[i]
        # x0,v0,v_l1_next = Q(N_diff[i])
        if i!=0:
            if strategy==1 or i!=1:
                with objmode(start1 = 'f8'):
                    start1 = time.perf_counter()
                x_f,x_c = correlated(dt_f,M_t,t0,T,eps,N_diff[i],Q,M,r,boundary=boundary,strategy=strategy)
                with objmode(end1 = 'f8'):
                    end1 = time.perf_counter()
            else:
                with objmode(start1 = 'f8'):
                    start1 = time.perf_counter()
                x_f,x_c = correlated(dt_f,round(levels[0]/dt_f),t0,T,eps,N_diff[i],Q,M,r,boundary=boundary,strategy=strategy)
                with objmode(end1 = 'f8'):
                    end1 = time.perf_counter()
            est_f = F(x_f);est_c=F(x_c)
            C_temp = (end1-start1)/N_diff[i]
            E_temp = np.mean(est_f-est_c)
            SS_temp = np.sum((est_f-est_c-E_temp)**2)
        else:
            with objmode(start2 = 'f8'):
                start2 = time.perf_counter()
            x = mc(dt_f,t0,T,N_diff[i],eps,Q,M,r,boundary=boundary)
            with objmode(end2 = 'f8'):
                end2 = time.perf_counter()
            est = F(x)
            C_temp = (end2-start2)/N_diff[i]
            E_temp = np.mean(est)
            SS_temp = np.sum((est-E_temp)**2)
        Delta = delta(E[i],E_temp,N[i],N_diff[i])
        E[i] = x_hat(N[i],N_diff[i],E[i],E_temp,Delta)
        SS[i] = Sfunc(N[i],N_diff[i],SS[i],SS_temp,Delta)
        '''Update cost like updating an average'''
        C[i] = x_hat(N[i],N_diff[i],C[i],C_temp,delta(C[i],C_temp,N[i],N_diff[i]))
        N[i] = N[i] + N_diff[i]
    return E,SS,N,N_diff,C


# @njit(nogil=True)
def ml(e2,Q,t0,T,M_t,eps,M,r,F,N_warm=40,boundary=None,strategy=1):
    '''
    e2: bound on mean square error
    Q: initial distribution
    t0: starting time
    T: end time
    M_t: dt_c/dt_f, where dt_c is coarse step size and dt_f is fine step size
    eps: mean free path, i.e. epsilon in model
    M: equilibrium distribution for velocity
    r: collision rate
    F: function used to find quantity of interest, E(F(X,V)).
    '''
    levels,N,N_diff,E,SS,C = select_levels(t0,T,M_t,eps,F,N=N_warm,strategy=strategy)
    '''
    levels: a list of step sizes for each level
    N: list of number of paths used
    N_diff: list of number of paths needed at every level
    E: estimation of quantity of interest at each level
    SS: sum of squares at each level
    C: Cost at each level
    '''
    L = len(levels)
    # C = M_t**np.arange(0,L,1); C[1:] = C[1:] + M_t**np.arange(0,L-1,1)
    '''While loop to continue until RMSE fits with e2'''
    while True:
        '''Update paths based on N_diff'''
        while np.max(N_diff)>0:
            I = np.where(N_diff > 0)[0] #Index for Levels that need more paths
            N_diff = np.minimum(N_diff,np.ones(len(N_diff),dtype=np.int64)*50_000)
            # print(f'index where more paths are needed: {I}, N_diff: {N_diff}')
            E,SS,N,N_diff,C = update_paths(I,E,SS,C,N,N_diff,levels,t0,T,M_t,eps,Q,M,r,F,boundary,strategy)
            V = SS/(N-1) #Update variance
            '''Determine number of paths needed with new information'''
            N_diff = np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64) - N
        '''Test bias is below e2/2'''
        test = max(abs(0.5*E[L-2]),abs(E[L-1])) < np.sqrt(e2/2)
        if test:
            break
        L += 1;
        # print(f'New level: {L}')
        N_diff = np.append(N_diff,N_warm).astype(np.int64)
        # with objmode():
        #     print(N_diff)
        N = np.append(N,0).astype(np.int64)
        E = np.append(E,0.0); V = np.append(V,0.0); SS = np.append(SS,0.0)
        C = np.append(C,0.0);#np.append(C,M_t**(L-1)+M_t**(L-2)); #
        if strategy==1:
            levels += [levels[0]/(M_t**(L-1))]
        else:
            levels += [levels[1]/(M_t**(L-2))]
    return E,V,C,N,levels

jit_module(nopython=True,nogil=True)
