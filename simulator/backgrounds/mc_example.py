from . import configlib
from .configlib import config as C
# import argparse
import numpy as np
from numba import jit,njit,jit_module,prange


'''Example to check consistency of Monte Carlo implementationss
Example is from:
"Asymptotic-PreservingScheme  Based  on  a  Finite  Volume/Particle-In-Cell
Coupling  for  Boltzmann-BGK-LikeEquations  in  the  Diffusion  Scaling", by Crestetto, Crouseilles and Lemou

 "Relaxed micro–macro schemes for kinetic equations" by Lemou'''

parser = configlib.add_parser('Background config')
parser.add_argument('--eps',default=1, help='epsilon', type=float)

# configlib.parse()
# epsilon = C['eps']
# print(epsilon)

# parser = argparse.ArgumentParser()
# parser.add_argument('eps', help='epsilon', type=float)
# args = parser.parse_args()

# epsilon = args.eps



'''Relevant methods for KD scheme'''
def S(N=10_000):
    '''
    N: number of samples
    '''
    #Distribution of interest in [0,1]
    rho = lambda x: 1+np.cos(2*np.pi*(x+0.5))
    #Proposal distribution - uniform in [0,1]
    q = lambda y: 1
    '''Samples, y, from q are accepted with probability rho(y)/(c*q(y)) where
    c = max(rho(x)/q(x))'''
    c = 2 #Determined before-hand
    rq = lambda n: np.random.uniform(0,1,size=n)
    #Variable to follow rho(x)
    X = np.zeros(N)
    #Indexes where proposals have NOT been accepted
    I = np.ones(N)
    n = np.sum(I,dtype=np.int64)
    while n>0:
        #Draw from proposal distribution
        Y = rq(n)
        #Draw uniform number
        U = np.random.uniform(0,1,size=n)
        #Test if reject or accept
        I_n = U <= rho(Y)/(c*q(Y))
        index_accept = np.where(I)[0][I_n]
        X[index_accept] = Y[I_n]
        I[index_accept] = 0
        n = np.sum(I,dtype=np.int64)
    v = np.random.uniform(-1,1,size=N)/epsilon
    v_norm = v/sigma(x)
    return X,v,v_norm

def R(x):
    epsilon = C['eps']
    return 1/(epsilon**2)

def dR(x):
    return 0

#Anti derivative of R
def R_anti(x):
    return x/(epsilon**2)


'''Relevant methods for splitting schemes'''
def S_nu(N=10_000):
    '''
    N: number of samples
    '''
    #Distribution of interest in [0,1]
    rho = lambda x: 1+np.cos(2*np.pi*(x+0.5))
    #Proposal distribution - uniform in [0,1]
    q = lambda y: 1
    '''Samples, y, from q are accepted with probability rho(y)/(c*q(y)) where
    c = max(rho(x)/q(x))'''
    c = 2 #Determined before-hand
    rq = lambda n: np.random.uniform(0,1,size=n)
    #Variable to follow rho(x)
    X = np.zeros(N)
    #Indexes where proposals have NOT been accepted
    I = np.ones(N)
    n = np.sum(I,dtype=np.int64)
    while n>0:
        #Draw from proposal distribution
        Y = rq(n)
        #Draw uniform number
        U = np.random.uniform(0,1,size=n)
        #Test if reject or accept
        I_n = U <= rho(Y)/(c*q(Y))
        index_accept = np.where(I)[0][I_n]
        X[index_accept] = Y[I_n]
        I[index_accept] = 0
        n = np.sum(I,dtype=np.int64)
    v = np.random.uniform(-1,1,size=N)
    v_norm = v


    return X,v,v_norm

def r(x):
    return np.ones(x.size)

jit_module(nopython=True,nogil=True)

if __name__ == '__main__':
    print(R(np.zeros(10)))
