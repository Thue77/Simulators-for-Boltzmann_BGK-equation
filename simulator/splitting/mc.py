from .one_step import phi_APS,phi_standard,phi_APS_new
from .correlated import correlated
from .AddPaths import delta,x_hat,Sfunc
import numpy as np
from numba import njit,jit_module
from numba import prange
import time
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


@njit(nogil=True)
def mc(dt,t0,T,N,eps,Q,M,r,boundary = None,sigma=None,rev=False,diff=False,v_ms=1,x0=None,v0=None):
    t = t0
    if x0 is None and v0 is None:
        x,v,_ = Q(N)
    else:
        x = x0.copy();v=v0.copy()
    while t<T:
        active = t<T
        z = np.random.normal(0,1,size=N); u = np.random.uniform(0,1,size=N)
        if rev: x,v,_ = phi_APS_new(x,v,dt,eps,z,u,M,r=r,boundary=boundary,diff=diff)
        else:
            x,v,_ = phi_APS(x,v,dt,eps,z,u,M,r=r,boundary=boundary,diff=diff)
        t += dt
    return x


def mc_adaptive(dt0,M_t,e2,N0,t0,T,N,eps,Q,M,r,F,boundary = None,density=False):
        dt=dt0
        bias = np.zeros(3)
        variance = 0
        N_total = N0
        '''To update sum of squares and mean when adding paths'''
        SS = 0; E=0;
        '''Run first three levels to ensure proper bias behaviour'''
        for i in range(3):
            dt_f = dt/M_t
            x_f,x_c = correlated(dt_f,M_t,t,T,eps,N0,Q,B,r,boundary=None,strategy = 1)
            bias[i] = np.mean(F(x_f)-F(x_c))
        pa = lin_fit(np.arange(3),np.log2(np.abs(bias))); alpha = -pa[0]
        test_bias = np.max(np.abs(bias[-3:])/M_t**(np.flip(np.arange(0,3))*alpha))/(M_t**alpha-1) < np.sqrt(e2/2)
        test_var = np.var(F(x_f)) < e2/2
        test = test_bias and test_var
        if density: X = x_f
        while not test:
            while not test_var:
                N = int(np.minimum(50_000,np.max(2*variance/e2-N_total,2)))
                x_f,x_c=correlated(dt_f,M_t,t0,T,eps,N,Q,M,r,boundary=boundary)
                if density: X = np.append(X,x_f)
                '''Update mean and variance of quantity of interes'''
                E_temp = np.mean(F(x_f))
                Delta = delta(E,E_temp,N_total,N)
                E = x_hat(N_total,N,E,E_temp,Delta)
                SS = Sfunc(N_total,N,SS,np.sum((F(x_f)**2)),Delta)
                variance = SS/(N_total-1)
                '''Update bias as well'''
                E_temp = np.mean(F(x_f)-F(x_c))
                Delta = delta(bias[-1],E_temp,N_total,N)
                bias[-1] = x_hat(N_total,N,bias[-1],E_temp,Delta)
                SS = Sfunc(N_total,N,SS,np.sum(((F(x_f)-F(x_c))**2)),Delta)
                '''Test variance'''
                test_var = variance < e2/2
                N_total += N
            pa = lin_fit(np.arange(3),np.log2(np.abs(bias[-3:]))); alpha = -pa[0]
            test_bias = np.max(np.abs(bias[-3:])/M_t**(np.flip(np.arange(0,3))*alpha))/(M_t**alpha-1) < np.sqrt(e2/2)
            '''If new level is needed, then the estimates are reset'''
            if not test_bias:
                dt_f=dt_f/M_t
                N=N0; N_total = N0
                bias = np.append(bias,0)
                x_f,x_c=correlated(dt_f,M_t,t0,T,eps,N,Q,M,r,boundary=boundary)
                if density:X = x_f
                E = np.mean(F(x_f)); SS = np.sum((F(x_f))**2)
                test_var = SS/(N_total-1) < e2/2
            pa = lin_fit(np.arange(3),np.log2(np.abs(bias[-3:]))); alpha = -pa[0]
            test_bias = np.max(np.abs(bias[-3:])/M_t**(np.flip(np.arange(0,3))*alpha))/(M_t**alpha-1) < np.sqrt(e2/2)
            test = test_var and test_bias
        if density:
            E,X
        else:
            return E



@njit(nogil=True)
def mc_standard(dt,t0,T,N,eps,Q,M,boundary,r,x0=None,v0=None):
    t = t0
    if x0 is None and v0 is None:
        x,v,_ = Q(N)
    else:
        x = x0.copy();v=v0.copy()
    while t<T:
        active = t<T
        # z = np.random.normal(0,1,size=N); u = np.random.uniform(0,1,size=N)
        x,v = phi_standard(x,v,dt,eps,M,boundary=boundary,r=r)
        t += dt
    return x


def lin_fit(x,y):
    p=1; n = x.size
    X = np.stack((np.ones(n),x),axis=1)
    normal_matrix = np.matmul(X.T,X)
    moment_matrix = np.matmul(X.T,y)
    return np.matmul(np.linalg.inv(normal_matrix),moment_matrix)[1]

def rnd1(x, decimals, out):
    return np.round_(x, decimals, out).astype(np.int64)


jit_module(nopython=True,nogil=True)


@njit(nogil=True,parallel=True)
def mc1_par(dt,t0,T,N,eps,Q,M,boundary,r,rev=False,diff=False,v_ms=1,x0=None,v0=None):
    cores = 8
    n = round(N/cores)
    x_AP = np.empty((cores,n))
    for i in prange(cores):
        if x0 is None and v0 is None:
            x_AP[i,:] = mc(dt,t0,T,n,eps,Q,M,r,boundary = boundary,rev=rev,diff=diff,v_ms=v_ms)
        else:
            x_AP[i,:] = mc(dt,t0,T,n,eps,Q,M,r,boundary = boundary,rev=rev,diff=diff,v_ms=v_ms,x0=x0[i*n:(i+1)*n],v0=v0[i*n:(i+1)*n])
    return x_AP.flatten()

@njit(nogil=True,parallel=True)
def mc2_par(dt,t0,T,N,eps,Q,M,boundary,r,x0=None,v0=None):
    cores = 8
    n = round(N/cores)
    x_std = np.empty((cores,n))
    for i in prange(cores):
        if x0 is None and v0 is None:
            x_std[i,:] = mc_standard(dt,t0,T,n,eps,Q,M,boundary,r)
        else:
            x_std[i,:] = mc_standard(dt,t0,T,n,eps,Q,M,boundary,r,x0=x0[i*n:(i+1)*n],v0=v0[i*n:(i+1)*n])
    return x_std.flatten()

def mc_density_test(dt_list,M_t,t0,T,N,eps,Q,M,r,F,boundary = None, x_std = None,rev=False,diff=False,v_ms=1,x0=None,v0=None):
    '''Returns a wasserstein distance and the associated standard deviation'''
    W_out = np.zeros(dt_list.size); err = np.zeros(dt_list.size); cost = np.zeros(dt_list.size)
    if x0 is None and v0 is None:
        x0,v0,_ = Q(N)
    if x_std is None: x_std = mc2_par((T-t0)/2**20,t0,T,N,eps,Q,M,boundary,r,x0=x0,v0=v0)
    for j,dt in enumerate(dt_list):
        W = np.zeros(20)
        print(dt)
        start = time.perf_counter()
        for i in range(20):
            x_AP = mc1_par(dt,t0,T,N,eps,Q,M,boundary,r,rev=rev,diff=diff,v_ms=v_ms,x0=x0,v0=v0)
            W[i] = wasserstein_distance(x_AP,x_std)
        cost[j] = time.perf_counter()-start
        W_out[j] = np.mean(W); err[j] = np.std(W)
    return W_out,err,cost
