from kinetic_diffusion.one_step import phi_KD,__psi_k
from kinetic_diffusion.mc import KDMC,Kinetic
from kinetic_diffusion.correlated import correlated as KD_C
from kinetic_diffusion.correlated import set_last_nonzero_col
from kinetic_diffusion.ml import warm_up,select_levels,select_levels_data
from kinetic_diffusion.ml import ml as KDML
from splitting.mc import mc as APSMC
from splitting.one_step import phi_SS
from splitting.correlated import correlated as AP_C
from splitting.correlated import correlated_test as AP_C_test
from accept_reject import test1,test2
from space import Omega
from AddPaths import Sfunc,delta,x_hat
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable,Tuple
import numpy as np
import pandas as pd
import math
from numba import njit,jit_module,prange
import time
import sys
import os
np.seterr(all='raise')


epsilon = float(sys.argv[1])#1
type = str(sys.argv[2])#'B1'
a = float(sys.argv[3])#5
b=float(sys.argv[4])#100
test = str(sys.argv[5])#'figure 5'
N_global = int(sys.argv[6])
print(f'a={a}, b={b},type={type}, test={test}')


'''Methods giving the properties of the plasma'''

#Inintial distribution of position and velocity
def Q(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Initial distribution for (x,v)'''
    if test == 'figure 4':
        # np.random.seed(4208)
        U = np.random.uniform(0,1,size=N)
        I = np.random.uniform(0,1,size=N)>0.5
        index = np.argwhere(I).flatten()
        index_not = np.argwhere(np.logical_not(I)).flatten()
        x = np.zeros(N)
        v = np.zeros(N)
        v_norm = np.append(np.random.normal(0,1,size = len(index)),np.random.normal(0,1,size = N-len(index)))
        v[index] = (v_norm[0:len(index)] + 10)
        v[index_not] = (v_norm[len(index):]-10)
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML' or test=='APS':
        x = np.ones(N)
        # print('Her')
        v_norm = np.random.normal(0,1,size=N)
        v = mu(x) + sigma(x)*v_norm
    elif test == 'num_exp_hom':
        x = test1(N); v = np.random.uniform(-1,1,size=N)/epsilon
        # x = np.zeros(N); v = np.zeros(N)
        # U = np.random.uniform(size = int(N/3))
        # x[:int(2*N/3)] = np.random.uniform(-0.5,0.5,size=int(2*N/3))
        # x[int(2*N/3):] = (U<=0.5)* np.random.uniform(-1.0,-0.5,size=int(N/3)) + (U>0.5)* np.random.uniform(0.5,1.0,size=int(N/3))
        # U = np.random.uniform(size = int(N/3))
        # v[:int(2*N/3)] = np.random.uniform(-0.75,0.25,size=int(2*N/3))
        # # print((U<=0.5)*np.random.uniform(-1.0,-0.75,size=int(N/3)))
        # v[int(2*N/3):] = (U<=0.25)*np.random.uniform(-1.0,-0.75,size=int(N/3)) + (U>0.25)* np.random.uniform(0.25,1.0,size=int(N/3))
        v_norm = v/sigma(x)
    elif test == 'num_exp':
        x,v,v_norm = test2(N)
        v = v/epsilon;v_norm=v_norm/epsilon
    return x,v,v_norm

def Q_nu(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Initial distribution for (x,nu)'''
    if type == 'default':
        x = test1(N); v = np.random.uniform(-1,1,size=N)
        v_norm = v.copy()
    elif type == 'Goldstein-Taylor':
        U = np.random.uniform(0,1,size=N)
        v =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)
        x = np.zeros(N)
        v_norm = v.copy()
    elif test == 'num_exp':
        x,v,v_norm = test2(N)
    return x,v,v_norm

#sets the collision rate
def R(x,alpha=0,beta=1):
    if type == 'default':
        return 1/(epsilon**2)
    elif type == 'B1':
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)
    elif type=='A':
        return 1/epsilon**2*(alpha*x+beta)

def dR(x,alpha=0,beta=1):
    if type == 'default':
        return 0
    elif type == 'B1':
        return (x<=1)*(-b*a) + (x>1)*(b*a)
    elif type=='A':
        if alpha==0:
            return 0
        else:
            return alpha/epsilon**2

#Anti derivative of R
def R_anti(x,alpha=0,beta=1):
    if type == 'default':
        return x/(epsilon**2)
    elif type == 'B1':
        return (-b*a/2*x**2 + (a+1)*b*x)*(x<=1) + (b*a/2*x**2+(1-a)*b*x)*(x>1)
    elif type=='A':
        return 1/epsilon**2*(alpha/2*x**2+beta*x)



#Sample Collision
# @njit(nogil=True,parallel = True)
def SC(x,v,e,alpha=0,beta=1):
    if type == 'default' or a==0:
        dtau = 1/R(x)*e
    elif (type == 'B1' or type == 'B2') and a!=0:
        '''The background is piece wise linear and it is necessary to identify
            particles that might move accross different pieces, e.g. if
            x>1 and v<0 then the particle might move below 1 where the collision
            rate is different'''
        n = len(x)
        boundaries = np.array([-math.inf,1,math.inf],np.float64) #Bins: (-inf,1] and (1,inf]
        num_of_bins = boundaries.size-1
        index = np.arange(0,n,dtype=np.int64)
        dtau = np.zeros(n)
        #The bin is given by the index of the lower boundary
        '''Indicates which domain each particle belongs to. It is given by the
            index of the last boundary that is smalle than x'''
        bins = np.array([np.where(p < boundaries)[0][0] for p in x],dtype=np.int64)-1
        direction = (v>0).astype(np.int64)-(v<0).astype(np.int64)#np.sign(v).astype(np.int64)
        slopes = np.array([-b*a,b*a],dtype=np.float64) if type=='B1' else np.array([0,b*a],dtype=np.float64)
        intercepts = np.array([(a+1)*b,(1-a)*b],dtype=np.float64) if type == 'B1' else np.array([b,(1-a)*b],np.float64)
        '''Need to subtract previous integrals from exponential number
            in each iteration and solve for the remainder. Note the multiplication
            by v. This is because the integration is done w.r.t. x an not t.'''
        e_remainder = np.abs(e*v) #The remainder of e after crossing into new domain
        e_local = e.copy()
        '''Need to update position of particle after crossing into new domain'''
        x_new = x.copy()
        while len(index)>0:
            '''Determine which bin each particle belongs to, given by index of upper bound'''
            '''Calculate integral within current bin'''
            I = integral_to_boundary(x_new[index],bins[index],direction[index],slopes,intercepts)
            index_new_domain = np.argwhere(I <= e_remainder[index]).flatten() #Index for particles that cross into different background
            index_new = index[index_new_domain] #In terms of original number of particles
            index_same_domain = np.argwhere(I > e_remainder[index]).flatten()
            index_same = index[index_same_domain]
            alpha = slopes[bins[index_same]];beta = intercepts[bins[index_same]]
            dtau[index_same] = dtau[index_same] + (-alpha*x_new[index_same]-beta + np.sqrt((alpha*x_new[index_same]+beta)**2+2*alpha*v[index_same]*e_local[index_same]))/(alpha*v[index_same])
            '''If the particle crosses into a new domain, the time it takes to cross
                needs to be added to the tau calculated in the new domain'''
            dtau[index_new] = dtau[index_new] + (1-x[index_new])/v[index_new]
            index = index_new.copy()
            # direction = direction[index_new_domain]
            bins[index_new] = bins[index_new] + direction[index_new]
            #location becomes the bound of the new domain
            '''Need to subtract the integral in the old domain from the exponential
                number'''
            e_remainder[index_new] = e_remainder[index_new] - I[index_new_domain]
            e_local[index_new] = e_local[index_new]-I[index_new_domain]/np.abs(v[index_new])
            '''Update x to equal the value of the boundary that it is crossing'''
            x_new[index_new] = boundaries[bins[index_new] + (direction[index_new]<0)]
    elif type=='A':
        if alpha==0:
            dtau = 1/R(x)*e
        else:
            dtau = (-alpha*x-beta + np.sqrt((alpha*x+beta)**2+2*alpha*v*epsilon**2*e))/(alpha*v)
    return dtau

@njit(nogil=True,parallel = True)
def integral_to_boundary(x,bins,direction,slopes,intercepts):
    '''
    Calculates integral of R to the boundary in the direction of the velocity.
    only works for B1 and B2. Integral is always positive. The affect of the
    sign of v has been removed


    boundaries: numpy array of boundaries for all domains
    slopes: numpy array of slope in each domain
    intercepts: numpy array of intercepts in each domain
    '''
    #Find function value where the domain changes
    B = slopes*1 + intercepts
    #index indicating finite integrals
    index = np.argwhere((bins==1)*(direction<0) + (bins==0)*(direction>0)).flatten()
    I = np.ones(len(x))*math.inf
    for j in prange(len(index)):
        i = index[j]
        I[i] = abs(x[i]-1)*B[bins[i]]+(abs(x[i]-1)*(slopes[bins[i]]*x[i]+intercepts[bins[i]]-B[bins[i]]))/2
    return I


def mu(x):
    if test == 'figure 4' or test=='APS':
        return 0
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML' or 'num_exp':
        return 0
    elif test == 'num_exp_hom':
        return 0#1/(2*epsilon)


def sigma(x):
    if test == 'figure 4' or test=='APS' or 'num_exp':
        return 1/epsilon
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML':
        return 1
    elif test == 'num_exp_hom':
        return np.sqrt(1/3)/epsilon

def M(x):
    '''Distribution of velocity scaled by epsilon, i.e. v'''
    if test == 'num_exp_hom':
        # v_norm = np.random.uniform(0,1)
        # v_next = (1-v_norm*2)/epsilon
        v_next = np.random.uniform(-1,1,size=len(x))/epsilon
        v_norm = v_next/sigma(x)
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm
    return v_next,v_norm

def M_nu(x):
    '''Distribution of velocity NOT scaled by epsilon, i.e. nu'''
    if test == 'num_exp_hom':
        # v_norm = np.random.uniform(0,1)
        # v_next = (1-v_norm*2)/epsilon
        v_next = np.random.uniform(-1,1,size=len(x))
        v_norm = v_next.copy()
    elif type == 'Goldstein-Taylor':
        U = np.random.uniform(0,1,size=len(x))
        v_next =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)
        v_norm = v_next.copy()
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm
    return v_next,v_norm


@njit(nogil=True)
def boundary_periodic(x):
    x0 = 0; xL = 1#/epsilon
    l = xL-x0 #Length of x domain
    I_low = (x<x0); I_high = (x>xL);
    x[I_low] = x[I_low] + l#xL-((x0-x[I_low])%l)
    x[I_high] = x[I_high] - l#x0 + ((x[I_high]-xL)%l)
    return x



jit_module(nopython=True,nogil=True)

'''Tests'''

def KDMC_test_fig_4_one_step(N,epsilon):
        BG = Plasma(mu=lambda x:0,sigma=lambda x:1/epsilon,epsilon=epsilon)
        np.random.seed(21)
        x0,v0,_ = BG.Q(N)
        dt = 1
        t = 0
        e = np.random.exponential(scale=1,size=N)
        tau = BG.SC(x0,v0,e)
        I_kd = tau < dt #Indicates particles that need to move both kinetically and diffusively
        index_kd = np.argwhere(I_kd).flatten()
        I_k = np.logical_not(I_kd) #Indicates particles that only need to move kinetically
        index_k = np.argwhere(I_k).flatten()
        xi = np.random.normal(size=index_kd.size)
        x_kd,v_kd,t_kd = phi_KD(dt,x0[index_kd],v0[index_kd],t,tau[index_kd],xi,BG.mu,BG.sigma,BG.M,BG.R)
        x_k,t_k = __psi_k(dt,x0[index_k],v0[index_k],t)
        # sns.histplot(x,bins=30)
        x = np.append(x_kd,x_k)
        dist = pd.DataFrame(data={'x':x})
        # tips = sns.load_dataset("tips")
        # print(type(tips))
        sns.kdeplot(data=dist, x="x")
        plt.show()

def KDMC_test_fig_4(N):
    '''
    epsilon = 0.1, 1 or 10
    type = 'default'
    test = 'figure 4'
    '''
    # BG = Plasma(mu=lambda x:0,sigma=lambda x:1/epsilon,epsilon=epsilon)
    x0,v0,_ = Q(N)
    dt = 1
    t = 0
    T = 1
    e = np.random.exponential(scale=1,size=N)
    tau = SC(x0,v0,e)
    # print(f'tau > T: {np.where(tau>T)}')
    start = time.time()
    x = KDMC(dt,x0,v0,e,tau,0,T,mu,sigma,M,R,SC)
    print(f'Before compile: {time.time()-start}')
    start = time.time()
    x = KDMC(dt,x0,v0,e,tau,0,T,mu,sigma,M,R,SC)
    print(f'After compile: {time.time()-start}')
    dist = pd.DataFrame(data={'x':x})
    sns.kdeplot(data=dist, x="x")
    plt.show()

def KD_cor_test_fig_4(N):
    '''Test for kinetic-diffusion correlated method to see distribution after 1 and 2 steps.
        Compare with figure 4 in MC article'''
    '''
    epsilon = 0.1, 1 or 10
    type = 'default'
    test = 'figure 4'
    '''
    x0,v0,v_l1_next = Q(N)
    x_f,x_c = KD_C(0.5,1,x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC)
    dist = pd.DataFrame(data={'x':x_c})
    sns.kdeplot(data=dist, x="x")
    plt.show()

@njit(nogil=True,parallel=True)
def KDML_cor_test_fig_5(N):
    '''
    epsilon is irrelevant
    type = 'B1' or 'B2'
    test = 'figure 5'
    set a and b as desired
    '''
    cache = 10_000
    dt_list = 1/2**np.arange(0,22,1)
    runs = int(max(N/cache,1))
    n = cache#int(N/cache) if N>cache else N
    V = np.zeros((runs,len(dt_list))); E = np.zeros((runs,len(dt_list)))
    V_d = np.zeros((runs,len(dt_list)-1)); E_bias = np.zeros((runs,len(dt_list)-1))
    if runs >1:
        for r in prange(runs):
            x0,v0,v_l1_next = Q(n)
            for i in prange(len(dt_list)-1):
                x_f,x_c = KD_C(dt_list[i+1],dt_list[i],x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR)
                E_bias[r,i] = np.mean(x_f-x_c)
                V_d[r,i] = np.sum((x_f-x_c-E_bias[r,i])**2)
                E[r,i] = np.mean(x_c)
                V[r,i] = np.sum((x_c-E[r,i])**2)
                if i == len(dt_list)-2:
                    E[r,i+1] = np.mean(x_f)
                    V[r,i+1] = np.sum((x_f-E[r,i+1])**2)
        V_out,_ = add_sum_of_squares_alongaxis(V,E,cache)
        V_d_out,_ = add_sum_of_squares_alongaxis(V_d,E_bias,cache)
        return V_out/(N-1),V_d_out/(N-1)
    else:
        x0,v0,v_l1_next = Q(N)
        for i in prange(len(dt_list)-1):
            # print(dt_list[i])
            x_f,x_c = KD_C(dt_list[i+1],dt_list[i],x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC,R_anti=R_anti)
            V_d[0,i] = np.var(x_f-x_c)#np.sum((x_f-x_c-np.mean(x_f-x_c))**2)
            V[0,i] = np.var(x_c)#np.sum((x_c-np.mean(x_c))**2)
            if i == len(dt_list)-2: V[0,i+1] = np.var(x_f)#np.sum((x_f-np.mean(x_f))**2)
        return V[0,:],V_d[0,:]


@njit(nogil=True,parallel=True)
def APML_cor_test_fig_5(N):
    '''
    epsilon is irrelevant
    type = 'B1' or 'B2'
    test = 'figure 5'
    set a and b as desired
    '''
    M_t = 2; t=0;T=5
    cache = 10_000
    dt_list = 2.5/M_t**np.arange(0,17,1)
    runs = int(max(N/cache,1))
    n = cache#int(N/cache) if N>cache else N
    V = np.zeros((runs,len(dt_list)-1)); E = np.zeros((runs,len(dt_list)-1))
    V_d = np.zeros((runs,len(dt_list)-1)); E_d = np.zeros((runs,len(dt_list)-1))
    if runs >1:
        for r in prange(runs):
            # x0,v0,v_l1_next = Q(n)
            for i in range(len(dt_list)-1):
                # print(dt_list[i])
                x_f,x_c = AP_C(dt_list[i+1],M_t,0,T,epsilon,n,Q_nu,M_nu)
                E_d[r,i] = np.mean(x_f**2-x_c**2)
                V_d[r,i] = np.sum((x_f**2-x_c**2-E_d[r,i])**2)
                E[r,i] = np.mean(x_c**2)
                V[r,i] = np.sum((x_c**2-E[r,i])**2)
                # if i == len(dt_list)-2:
                #     E[r,i+1] = np.mean(x_f)
                #     V[r,i+1] = np.sum((x_f-E[r,i+1])**2)
        V_out,E_out = add_sum_of_squares_alongaxis(V,E,cache)
        V_d_out,E_d_out = add_sum_of_squares_alongaxis(V_d,E_d,cache)
        return V_out/(N-1),E_out,V_d_out/(N-1),np.abs(E_d_out)
        # return add_sum_of_squares_alongaxis(V,E,cache)/(N-1),add_sum_of_squares_alongaxis(V_d,E_bias,cache)/(N-1)
    else:
        for i in prange(len(dt_list)-1):
            # print(dt_list[i])
            x_f,x_c = AP_C(dt_list[i+1],M_t,0,T,epsilon,N,Q_nu,M_nu)
            V_d[0,i] = np.var(x_f**2-x_c**2)#np.sum((x_f-x_c-np.mean(x_f-x_c))**2)
            E_d[0,i] = np.mean(x_f**2-x_c**2)
            V[0,i] = np.var(x_c**2)#np.sum((x_c-np.mean(x_c))**2)
            E[0,i] = np.mean(x_c**2)
            # if i == len(dt_list)-2:
            #     V[0,i+1] = np.var(x_f)#np.sum((x_f-np.mean(x_f))**2)
            #     V[0,i+1] = np.mean(x_f)
        return V[0,:],E[0,:],V_d[0,:],np.abs(E_d[0,:])

@njit(nogil=True,parallel=True)
def add_sum_of_squares_alongaxis(A,A_mean,N,axis=0):
    '''If axis is zero then the sum of squares of each coulmn is calculated
    A: matrix of sum of squares for each run
    A_mean: matrix of means for each run
    N: number of paths used in each run
    '''
    n = A.shape[axis]
    m = A.shape[0] if axis==1 else A.shape[1]
    ss = A[0,:]
    E = np.zeros(m)
    for j in prange(m):
        mu_old = A_mean[0,j]
        # print(mu_old)
        for i in range(1,n):
            Delta = A_mean[i,j] - mu_old
            M = (N*(N*(i)))/(N+N*(i))
            ss[j] = ss[j] + A[i,j] + Delta**2*M
            mu_old = mu_old + Delta*(N*i)/(N+N*i)
        E[j] = mu_old
    return ss,E



def test_warm_up(N=100,L=21):
    L = 21; t0 = 0; T = 1
    Q_l,Q_l_L,V_l,V_l_L,C_l,C_l_L = warm_up(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N=N)
    plt.plot(1/2**np.arange(1,L+1),C_l_L/N)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def test_level_selection(plot=True):
    if plot:
        data = np.loadtxt(f'var_a_{a}_b_{b}_type_{type}.txt')
        V_before = data[0]
        V_d = data[1][:-1]
        dt_list = 1/2**np.arange(0,22)
        plt.plot(dt_list[1:],V_d,label='All levels')
        levels,V = select_levels_data(data)
    else:
        L = 21; t0 = 0; T = 1
        dt_list = 1/2**np.arange(1,L+1)
        levels,N,X,V,C = select_levels(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N=100,tau=None)
    plt.plot(dt_list[levels],V,'.',color='black',label='Selected levels')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('\u0394 t')
    plt.ylabel('variance')
    plt.legend()
    plt.show()
    print(f'levels: {levels}')

def test_num_exp_hom_MC():
    '''Plot results for numerical example based on radiative transport'''
    df = pd.DataFrame(data={'x':[],'Diffusion paramter':[]})
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    for eps in [0.01,0.5,1.0]:
        rel_path = f'num_exp_hom/APS_eps_{eps}.txt'
        abs_file_path = os.path.join(script_dir, rel_path)
        x = np.loadtxt(abs_file_path)
        df = df.append(pd.DataFrame(data={'x':x,'Diffusion paramter':[f'epsilon = {eps}' for _ in range(len(x))]}))
    sns.kdeplot(data=df, x="x",hue='Diffusion paramter',cut=0,linestyle='--',common_norm=False)
    '''Repeat for KDMC'''
    df = pd.DataFrame(data={'x':[],'Diffusion paramter':[]})
    for eps in [0.01,0.5,1.0]:
        rel_path = f'num_exp_hom/KDMC_eps_{eps}.txt'
        abs_file_path = os.path.join(script_dir, rel_path)
        x = np.loadtxt(abs_file_path)
        df = df.append(pd.DataFrame(data={'x':x,'Diffusion paramter':[f'epsilon = {eps}' for _ in range(len(x))]}))
    sns.kdeplot(data=df, x="x",hue='Diffusion paramter',cut=0,linestyle='dotted',common_norm=False)

def KDML_test():
    E,V,C,N,levels = KDML(1e-3,Q,0,1,mu,sigma,M,R,SC,R_anti,dR,tau=None,L=22,N_warm = 10)
    print(f'Multilevel result: {np.sum(E)}')
    print(N)
    print(f'levels: {levels}')
    print(f'variance at each level: {V}')
    print(f'variance: {np.sum(V/N)}')
    plt.plot(1/2**levels,V)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def Kinetic_test():
    N=N_global
    x = Kinetic(N,Q,0,1,mu,sigma,M,R,SC)
    print(f'Kinetic result: {np.mean(x)}')


def numerical_experiemnt():
    '''test=num_exp,type=A'''
    alpha = input('Give alpha: ')
    beta = input('Give beta: ')
    R = lambda x: 1/epsilon**2*(alpha*x+beta)
    KDMC(dt,x0,v0,e,tau,0,T,mu,sigma,M,R,SC)



'''Exists in separate file as well'''
def plot_var(V,V_d):
    dt_list = 1/2**np.arange(0,22,1)
    plt.plot(dt_list[:-1],V_d,':', label = f'a={a}')
    plt.plot(dt_list,V,'--',color = plt.gca().lines[-1].get_color())
    plt.title(f'b = {b}, type: {type}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

'''Probably redundant:'''
def update_a(b):
    global a
    a = a + b

'''Maybe helps with total recompilation'''
def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)

def kill_numba_cache():

    root_folder = os.path.realpath('C:/Users/thom1/OneDrive/SDU/Speciale/Programming/Package_version/simulator')

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)



if __name__ == '__main__':
    if test == 'figure 4' and type == 'default':
        K = input('Cor or MC?\n')
        if K == 'MC':
            KDMC_test_fig_4(500_000)
        else:
            KD_cor_test_fig_4(100_000)
    elif test == 'figure 5' and (type == 'B1' or type == 'B2'):
        print('Starting')
        KDML_cor_test_fig_5(10)
        start = time.time()
        V,V_d = KDML_cor_test_fig_5(N_global)
        print(f'elapsed time is {time.time()-start}')
        # np.savetxt(f'var_a_{a}_b_{b}_type_{type}.txt',np.vstack((V,np.append(V_d,0))))
        # print(f'V: {V}')
        plot_var(V,V_d)
    elif test == 'warm_up' and (type == 'B1' or type == 'B2'):
        test_warm_up()
    elif test == 'select_levels' and (type == 'B1' or type == 'B2'):
        test_level_selection()
    elif test == 'KDML' and (type == 'B1' or type == 'B2'):
        KDML_test()
        # Kinetic_test()
    elif test == 'num_exp_hom' and type=='default':
        '''Numerical experiemnt on radiative transport with
        rho(x,0) = 1+cos(2*pi*(x+0.5)) and V ~ U(-1,1)'''
        x_lim = (0,1); v_lim = (-1,1)
        N = 400_000
        dt = 0.005;t0=0;T=0.1
        # test_num_exp_hom_MC()
        if False:
            x = APSMC(dt,t0,T,N,epsilon,Q_nu,M_nu,boundary=boundary_periodic)
            # np.savetxt(f'APS_eps_{epsilon}.txt',x)
        elif True:
            x0,v0,_ = Q(N)
            # e = np.random.exponential(size=N); tau = SC(x0,v0,e)
            x = KDMC(dt,x0,v0,t0,T,mu,sigma,M,R,SC,boundary=boundary_periodic)
            np.savetxt(f'KDMC_eps_{epsilon}.txt',x)
        elif False:
            t = t0
            # dt = min(epsilon**2/2,dt)
            print(dt)
            x,_,_ = Q(N)
            v = epsilon/(epsilon**2+dt)*np.random.uniform(-1,1,size=len(x))
            # dist = pd.DataFrame(data={'x':x})
            # sns.kdeplot(data=dist, x="x",cut=0,label='Initial')
            while t<T:
                x,v = phi_SS(x,v,dt,epsilon)
                x = boundary_periodic(x)
                t += dt
        else:
            x = Kinetic(N,Q,t0,T,mu,sigma,M,R,SC,boundary=boundary_periodic)
        # print(x)
        # rho = SP.density_estimation(x)
        # x=x*epsilon
        dist = pd.DataFrame(data={'x':x})
        sns.kdeplot(data=dist, x="x",cut=0)
        # plt.plot(SP.x_axis,rho)
        plt.show()
    elif test == 'corr_path' and type=='Goldstein-Taylor':
        '''Plot the fine and coarse path of the APML method under the Goldstein-
        Taylor model to compare with article on APML. Set epsilon = 0.5'''
        dt_f = 0.2;M_t=5;t=0;T=10;N=1
        AP_C_test(dt_f,M_t,t,T,epsilon,N,Q_nu,M_nu,plot=True)
    elif test == 'var_path' and type=='Goldstein-Taylor':
        '''Plot the variance fine and coarse paths and their difference of the
        APML method under the Goldstein-Taylor model to compare with article on APML.
        Set epsilon = 0.5'''
        dt_f = 0.2;M_t=5;t=0;T=10;N=100_000
        AP_C_test(dt_f,M_t,t,T,epsilon,N,Q_nu,M_nu,plot_var=True)
    elif test == 'var_structure' and type=='Goldstein-Taylor':
        '''Plot the variance fine and coarse paths and their difference of the
        APML method under the Goldstein-Taylor model to compare with article on APML.
        Set epsilon = 10,1,0.1'''
        N=100_000
        M_t = 2
        dt_list = 2.5/M_t**np.arange(0,17,1)
        V,E,V_d,E_d=APML_cor_test_fig_5(N)
        '''Alternatively, run with loaded data from var_results directory by
        commenting the above line and uncommenting the below'''
        # script_dir = os.path.dirname(__file__)
        # rel_path = f'var_results/var_eps_{epsilon}_type_{type}.txt'
        # abs_file_path = os.path.join(script_dir, rel_path)
        # V,V_d = np.loadtxt(abs_file_path)
        # rel_path = f'var_results/E_eps_{epsilon}_type_{type}.txt'
        # abs_file_path = os.path.join(script_dir, rel_path)
        # E,E_d = np.loadtxt(abs_file_path)
        plt.figure(1)
        plt.subplot(122)
        x2 = dt_list[10]; x1 = dt_list[13]
        y2 = V_d[9]; y1 = V_d[12]
        slope = (math.log10(y2)-math.log10(y1))/(math.log10(x2)-math.log10(x1))
        x_end = 2e-3;x_start=1e-4
        plt.plot(x1+np.linspace(x_start,x_end,4),np.tile(y1,4),color='black') # plot horizontal line
        t = math.log10(y1) + slope*(math.log10(x1+x_end)-math.log10(x1+x_start)) # top of triangle
        y = np.linspace(y1,10**t,5)
        plt.plot(np.tile(x1+x_end,5),y,color='black') #plot vertical line
        plt.text(x1+(x_end-x_start)/2,y[0]+0.00625*(y[-1]-y[0]),f'{round(slope)}') #plot slope
        plt.plot(x1+np.linspace(x_start,x_end,4),(x1+np.linspace(x_start,x_end,4))**slope*(y1/(x1+x_start)**slope),color='black') #plot diagonal line
        plt.plot(dt_list[1:],V_d,label='Diff')
        plt.plot(dt_list[1:],V,'--',label='Single')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('\u0394 t')
        plt.ylabel('Variance')
        # plt.legend()
        plt.subplot(121)
        x2 = dt_list[10]; x1 = dt_list[13]
        y2 = E_d[9]; y1 = E_d[12]
        slope = (math.log10(y2)-math.log10(y1))/(math.log10(x2)-math.log10(x1))
        # y1 = y1-0.0038; #For epsilon=1
        x_end = 1e-3;x_start=1e-4
        plt.plot(x1+np.linspace(x_start,x_end,4),np.tile(y1,4),color='black') # plot horizontal line
        t = math.log10(y1) + slope*(math.log10(x1+x_end)-math.log10(x1+x_start)) # top of triangle
        y = np.linspace(y1,10**t,5)
        plt.plot(np.tile(x1+x_end,5),y,color='black') #plot vertical line
        plt.text(x1+(x_end-x_start)/2,y[0]+0.00625*(y[-1]-y[0]),f'{round(slope)}') #plot slope
        plt.plot(x1+np.linspace(x_start,x_end,4),(x1+np.linspace(x_start,x_end,4))**slope*(y1/(x1+x_start)**slope),color='black') #plot diagonal line
        plt.plot(dt_list[1:],E_d,label='Diff')
        plt.plot(dt_list[1:],E,'--',label='Single')
        plt.ylabel('Mean')
        plt.xlabel('\u0394 t')
        plt.xscale('log')
        plt.yscale('log')
        # plt.legend()
        plt.show()



    # print(test_numba(np.array([1,-2,3,4,-5],dtype=np.float64)))
