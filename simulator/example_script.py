from kinetic_diffusion.one_step import phi_KD,__psi_k
from kinetic_diffusion.mc import KDMC,Kinetic
from kinetic_diffusion.correlated import correlated as KD_C
from kinetic_diffusion.correlated import set_last_nonzero_col
from kinetic_diffusion.ml import warm_up,select_levels
from kinetic_diffusion.ml import ml as KDML
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
print(f'a={a}, b={b},type={type}')


'''Methods giving the properties of the plasma'''

def M(x):
    return np.random.normal(0,1,size=x.size)

#Inintial distribution of position and velocity
def Q(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
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
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML':
        x = np.ones(N)
        v_norm = np.random.normal(0,1,size=N)
        v = mu(x) + sigma(x)*v_norm
    return x,v,v_norm

#sets the collision rate
def R(x):
    if type == 'default':
        return 1/(epsilon**2)
    elif type == 'B1':
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)

def dR(x):
    if type == 'default':
        return 0
    elif type == 'B1':
        return (x<=1)*(-b*a) + (x>1)*(b*a)

#Anti derivative of R
def R_anti(x):
    if type == 'default':
        return x/(epsilon**2)
    elif type == 'B1':
        return (-b*a/2*x**2 + (a+1)*b*x)*(x<=1) + (b*a/2*x**2+(1-a)*b*x)*(x>1)

#Sample Collision
# @njit(nogil=True,parallel = True)
def SC(x,v,e):
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
    if test == 'figure 4':
        return 0
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML':
        return 0


def sigma(x):
    if test == 'figure 4':
        return 1/epsilon
    elif test == 'figure 5' or test == 'warm_up' or test == 'select_levels' or test == 'KDML':
        return 1


def test_numba(x):
    return np.sign(x).astype(np.int64)


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
                # print(dt_list[i])
                x_f,x_c = KD_C(dt_list[i+1],dt_list[i],x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR)
                E_bias[r,i] = np.mean(x_f-x_c)
                V_d[r,i] = np.sum((x_f-x_c-E_bias[r,i])**2)
                E[r,i] = np.mean(x_c)
                V[r,i] = np.sum((x_c-E[r,i])**2)
                if i == len(dt_list)-2:
                    E[r,i+1] = np.mean(x_f)
                    V[r,i+1] = np.sum((x_f-E[r,i+1])**2)
        return add_sum_of_squares_alongaxis(V,E,cache)/(N-1),add_sum_of_squares_alongaxis(V_d,E_bias,cache)/(N-1)
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
def compute_mean_alongaxis(A,axis=0):
    '''If axis is zero then the mean of each coulmn is calculated'''
    n = A.shape[axis]
    m = A.shape[0] if axis==1 else A.shape[1]
    mu = np.zeros(m)
    for j in prange(m):
        for i in prange(n):
            mu[j] = mu[j] + 1/(i+1)*(A[i,j]-mu[j])
    return mu

@njit(nogil=True,parallel=True)
def add_sum_of_squares_alongaxis(A,A_mean,N,axis=0):
    '''If axis is zero then the mean of each coulmn is calculated
    A: matrix of sum of squares for each run
    A_mean: matrix of means for each run
    N: number of paths used in each run
    '''
    n = A.shape[axis]
    m = A.shape[0] if axis==1 else A.shape[1]
    ss = np.zeros(m)
    for j in prange(m):
        mu_old = 0
        for i in prange(n):
            Delta = A_mean[i,j] - mu_old
            M = (N*(N*(i)))/(N+N*(i))
            ss[j] = ss[j] + A[i,j] + Delta**2*M
            mu_old = mu_old + Delta*(N*i)/(N+N*(i))
    return ss


def test_warm_up(N=100,L=21):
    L = 21; t0 = 0; T = 1
    Q_l,Q_l_L,V_l,V_l_L,C_l,C_l_L = warm_up(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N=N)
    plt.plot(1/2**np.arange(1,L+1),C_l_L/N)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def test_level_selection():
    # data = np.loadtxt(f'var_a_{a}_b_{b}_type_{type}.txt')
    L = 21; t0 = 0; T = 1
    levels,N,X,V,C = select_levels(L,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,N=100,tau=None)
    dt_list = 1/2**np.arange(1,L+1)
    # plt.plot(dt_list,V_d[:-1],':')
    plt.plot(dt_list[levels],V,':')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print(f'levels: {levels}')


def KDML_test():
    E,V,C,N,levels = KDML(1e-5,Q,0,1,mu,sigma,M,R,SC,R_anti,dR,tau=None,L=14,N_warm = 100)
    print(f'Multilevel result: {np.sum(E)}')
    print(N)
    print(f'variance: {np.sum(V/N)}')
    plt.plot(1/2**levels,V)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def Kinetic_test():
    N=N_global
    x = Kinetic(N,Q,0,1,mu,sigma,M,R,SC)
    print(f'Kinetic result: {np.mean(x)}')


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
    # os.environ['NUMBA_DISABLE_JIT'] = '1'
    # print(dir(one_step))
    # e = 2; x=5; v = 0;
    # print(f'Before recompile of R \n SC= {SC(x,v,e)}, R={R(x)}')
    # update_a(4)
    # SC.recompile()
    # print(f'After recompile of R \n SC= {SC(x,v,e)}, R={R(x)}')
    # a = 4
    # print(f'Outside: {R(10)}')
    if test == 'figure 4' and type == 'default':
        K = input('Cor or MC?\n')
        if K == 'MC':
            KDMC_test_fig_4(500_000)
        else:
            KD_cor_test_fig_4(100_000)
    elif test == 'figure 5' and type == 'B1' or type == 'B2':
        print('Starting')
        KDML_cor_test_fig_5(10)
        start = time.time()
        V,V_d = KDML_cor_test_fig_5(N_global)
        print(f'elapsed time is {time.time()-start}')
        np.savetxt(f'var_a_{a}_b_{b}_type_{type}.txt',np.vstack((V,np.append(V_d,0))))
        # print(f'V: {V}')
        # plot_var(V,V_d)
    elif test == 'warm_up' and (type == 'B1' or type == 'B2'):
        test_warm_up()
    elif test == 'select_levels' and (type == 'B1' or type == 'B2'):
        test_level_selection()
    elif test == 'KDML' and (type == 'B1' or type == 'B2'):
        KDML_test()
        Kinetic_test()


    # print(test_numba(np.array([1,-2,3,4,-5],dtype=np.float64)))
