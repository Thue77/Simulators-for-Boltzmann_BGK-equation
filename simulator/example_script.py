from plasma_example.plasma import Plasma
from kinetic_diffusion.one_step import phi_KD,__psi_k
from kinetic_diffusion.mc import KDMC
from kinetic_diffusion.correlated import correlated as KD_C
from kinetic_diffusion.correlated import set_last_nonzero_col
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable,Tuple
import numpy as np
import pandas as pd
from numba import njit
import time
import os

epsilon = 0.1
type = 'default'
a = 0; b=1
test = 'figure 4'

'''Tests'''

def KDMC_test_fig_4_one_step(N,epsilon):
        BG = Plasma(mu=lambda x:0,sigma=lambda x:1/epsilon,epsilon=epsilon)
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
    x_k1,x_k2 = KD_C(0.5,1,x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC)
    dist = pd.DataFrame(data={'x':x_k2})
    sns.kdeplot(data=dist, x="x")
    plt.show()

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

'''Methods giving the properties of the plasma'''
@njit
def M(x):
    return np.random.normal(0,1,size=x.size)

#Inintial distribution of position and velocity
@njit
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
        return x,v,v_norm

#sets the collision rate
@njit
def R(x):
    if type == 'default':
        return 1/(epsilon**2)
    elif type == 'B1':
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)

#Simpler version of SampleCollision
@njit
def SC(x,v,e):
    if type == 'default' or a==0:
        dtau = 1/R(x)*e
        return dtau

@njit
def mu(x):
    if test == 'figure 4':
        return 0

@njit
def sigma(x):
    if test == 'figure 4':
        return 1/epsilon





if __name__ == '__main__':
    # print(dir(one_step))
    # e = 2; x=5; v = 0;
    # print(f'Before recompile of R \n SC= {SC(x,v,e)}, R={R(x)}')
    # update_a(4)
    # SC.recompile()
    # print(f'After recompile of R \n SC= {SC(x,v,e)}, R={R(x)}')
    # a = 4
    # print(f'Outside: {R(10)}')
    # KDMC_test_fig_4(500_000)
