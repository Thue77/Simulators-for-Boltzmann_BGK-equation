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
import math
from numba import jit_module
import time
import os

epsilon = 1
type = 'B1'
a = 0; b=1
test = 'figure 5'


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
    elif test == 'figure 5':
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

#Simpler version of SampleCollision

def SC(x,v,e):
    if type == 'default' or a==0:
        dtau = 1/R(x)*e
    elif (type == 'B1' or type == 'B2') and a!=0:
        '''The background is piece wise linear and it is necessary to identify
            particles that might move accross different pieces, e.g. if
            x>1 and v<0 then the particle might move below 1 where the collision
            rate is different'''
        boundaries = np.array([-math.inf,1,math.inf]) #Bins: (-inf,1] and (1,inf]
        num_of_bins = np.size(boundaries)-1
        index = np.arange(0,n,dtype=int)
        dtau = np.zeros(n)
        bins = np.array([np.where(p <= boundaries)[0][0] for p in x],dtype=np.int64)
        direction = np.sign(v).astype(np.int64)
        slopes = np.array([-b*a,b*a]) if type=='B1' else np.array([0,b*a])
        intercepts = np.array([(a+1)*b,(1-a)*b]) if type == 'B1' else np.array([b,(1-a)*b])
        '''The lower bound of the integral changes. First it is x, but in
            succesive iterations it is the lower bound of the current bin'''
        location = x.copy()
        '''Need to subtract previous integrals from exponential number
            in each iteration and solve for the remainder'''
        e_remainder = e.copy() #The remainder of e after crossing into new domain
        '''Need to update position of particle after crossing into new domain'''
        x_new = x.copy()
        while len(index)>0:
            # print(f'e_r = {e_remainder[2]}')
            '''Determine which bin each particle belongs to, given by index of upper bound'''
            '''Calculate integral within current bin'''
            I = integral_to_boundary(location,bins,direction,slopes,intercepts)#np.array([quad(lambda x: self.get_slope(xk)*x + self.get_intercept(xk),xk,boundaries[b-d])[0]/vk if (b-d>0 and b-d<num_of_bins) else np.inf for xk,vk,b,d in zip(lb,v[index],bins,(direction==-1))])
            index_new_domain = np.argwhere(I <= e_remainder[index]).flatten() #Index for particles that cross into different background
            # print(f'index_new_domain = {index_new_domain}, \n index = {index}')
            index_new = index[index_new_domain] #In terms of original number of particles
            # print(f'index_new = {index_new}')
            index_same_domain = np.argwhere(I > e_remainder[index]).flatten()
            alpha = self.get_slope(boundaries[bins[index_same_domain]]); beta = self.get_intercept(boundaries[bins[index_same_domain]])
            # print(f'x1 = {x[1]}, x4 = {x[4]} \n 1 = {boundaries[bins[index_same_domain]][1]}, 2 = {boundaries[bins[index_same_domain]][2]}')
            # print(f'alpha1 = {alpha[1]}, alpha4 = {alpha[4]}')
            index_same = index[index_same_domain]
            alpha = slopes[bins];beta = intercepts[bins]
            dtau[index_same] = dtau[index_same] + (-alpha*x_new[index_same]-beta + np.sqrt((alpha*x_new[index_same]+beta)**2+2*alpha*v[index_same]*e_remainder[index_same]))/(alpha*v[index_same])
            dtau[index_new] = dtau[index_new] + (boundaries[bins[index_new_domain]-(direction[index_new_domain]==-1)]-x[index_new])/v[index_new]
            index = index_new.copy()
            direction = direction[index_new_domain]
            bins = bins[index_new_domain] + direction
            #location becomes the bound of the new domain
            location = boundaries[bins -1]
            e_remainder[index_new] = e_remainder[index_new] - I[index_new_domain]
            x_new[index_new] = boundaries[bins - (direction>0)]
    return dtau


def integral_to_boundary(x,bins,direction,slopes,intercepts):
    '''Calculates integral of R to the boundary in the direction of the velocity.
        only works for B1 and B2

        boundaries: numpy array of boundaries for all domains
        slopes: numpy array of slope in each domain
        intercepts: numpy array of intercepts in each domain
    '''
    #Find where the domain changes
    B = slopes*1 + intercepts
    #other bound of integral. The first bound is the position
    bounds = bins + (direction==1)
    #index to indicate finite integrals
    index = bounds == 1
    I = np.ones(len(x))*math.inf
    for i in index:
        I[i] = abs(x[i]-1)*B[i]+(abs(x[i]-1)*(slopes[bins[i]]*x[i]+intercepts[bins[i]]-B[bins[i]]))/2
    return I


def mu(x):
    if test == 'figure 4':
        return 0
    elif test == 'figure 5':
        return 0


def sigma(x):
    if test == 'figure 4':
        return 1/epsilon
    elif test == 'figure 5':
        return 1


def test_numba(x):
    return np.sign(x).astype(np.int64)

jit_module(nopython=True,nogil=True)

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
    x_f,x_c = KD_C(0.5,1,x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC)
    dist = pd.DataFrame(data={'x':x_c})
    sns.kdeplot(data=dist, x="x")
    plt.show()

def KDML_cor_test_fig_5(N):
    '''
    epsilon is irrelevant
    type = 'B1' or 'B2'
    test = 'figure 5'
    set a and b as desired
    '''
    V_d = []
    V = []
    dt_list = 1/2**np.arange(0,16,1)
    x0,v0,v_l1_next = Q(N)
    for i in range(len(dt_list)-1):
        x_f,x_c = KD_C(dt_list[i+1],dt_list[i],x0,v0,v_l1_next,0,1,mu,sigma,M,R,SC)
        print(dt_list[i])
        V_d += [np.var(x_f-x_c)]
        V += [np.var(x_c)]
        if i == len(dt_list)-2: V += [np.var(x_f)]
    plt.plot(dt_list[:-1],V_d,':', label = f'a={a}')
    plt.plot(dt_list,V,'--',color = plt.gca().lines[-1].get_color())
    plt.title(f'b = {b}, type: {type}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
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
    # KD_cor_test_fig_4(100_000)
    KDML_cor_test_fig_5(100)



    # print(test_numba(np.array([1,-2,3,4,-5],dtype=np.float64)))
