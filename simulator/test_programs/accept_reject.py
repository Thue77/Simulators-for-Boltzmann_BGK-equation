import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit

'''Script to generate samples from rho(x,0) via the accept-reject method'''

'''-----------First example--------------'''
@njit(nogil=True)
def test1(N=10_000):
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
    return X


if __name__ == '__main__':
    X = test1()
    rho = lambda x: 1+np.cos(2*np.pi*(x+0.5))
    dist = pd.DataFrame(data={'x':X})
    sns.kdeplot(data=dist, x="x")
    plt.plot(np.arange(0,1,0.001),rho(np.arange(0,1,0.001)),label='exact')
    plt.legend()
    plt.show()
