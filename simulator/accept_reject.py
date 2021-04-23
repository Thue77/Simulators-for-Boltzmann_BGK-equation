import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.stats as st
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

def plot_test(data_sample=None):
    '''Example on how to get density based on observations using kernel density
    estimation '''
    if data_sample is None:
        N = 100000
        data_sample = np.random.normal(0, 1, size=(2,N))
    # dx = 0.1
    # data_2d = np.zeros((len(y_values),len(x_values)))
    # for i,y in enumerate(y_values):
    #     for j,x in enumerate(x_values):
    #         data_2d[i,j] = np.sum(np.logical_and(data_sample[0,:]>=x-dx/2,data_sample[0,:]<=x+dx/2)*np.logical_and(data_sample[1,:]>=y-dx/2,data_sample[1,:]<=y+dx/2))
    # print(np.logical_and(data_sample[0,:]>=0-dx/2,data_sample[0,:]<=0+dx/2))
    # print(data_2d)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # X, Y = np.meshgrid(x_values, y_values)
    # surf = ax.plot_surface(X, Y, data_2d/N, cmap='coolwarm', edgecolor='none')
    # values = np.vstack([x_values, y_values])
    # print(values)
    # kernel = st.gaussian_kde(data_sample)
    # print(kernel(values))
    # f = np.reshape(kernel(values).T, X.shape)

    m1, m2 = data_sample[0,:],data_sample[1,:]
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    #Perform a kernel density estimate on the data:
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, X.shape)
    print(f'density value at 0,0: {kernel((0,0))}')
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of estimated f(x,v,t=0)')
    surf = ax.plot_surface(X, Y, f, rstride=1, cstride=1, edgecolor='none')

@njit(nogil=True)
def test2(N=10_000):
    '''Accept-reject method for f(x,v,t=0)= 1/sqrt(2*pi)*v^2*e^{-v^2/2}*(1+cos(2*pi*(x+1/2)))'''
    c = 2.9
    q = lambda x,v: 0.5*1/np.sqrt(2*np.pi)*(np.exp(-(v-np.sqrt(2))**2/2)+np.exp(-(v+np.sqrt(2))**2/2))
    f = lambda x,v:1/np.sqrt(2*np.pi)*np.exp(-v**2/2)*v**2*(1+np.cos(2*np.pi*(x+0.5)))
    #Function to generate random numbers from q
    def rq(n):
        x = np.random.uniform(0,1,size=n)
        #Uniform numbers for velocity distribution
        u_v = np.random.uniform(0,1,size=n)
        #positive mean
        I_p = u_v<=0.5
        #negative mean
        I_n = np.logical_not(I_p)
        v = np.zeros(n)
        v[I_p] = np.random.normal(np.sqrt(2),1,size=np.sum(I_p))
        v[I_n] = np.random.normal(-np.sqrt(2),1,size=np.sum(I_n))
        return x,v
    I = np.ones(N)
    n = np.sum(I,dtype=np.int64)
    X = np.zeros(N); V = np.zeros(N)
    while n>0:
        x,v = rq(n)
        #Uniform number to test whether to accept or reject
        U = np.random.uniform(0,1,size=n)
        #Test if reject or accept
        I_n = U <= f(x,v)/(c*q(x,v))
        index_accept = np.where(I)[0][I_n]
        X[index_accept] = x[I_n];V[index_accept] = v[I_n]
        I[index_accept] = 0
        n = np.sum(I,dtype=np.int64)
    return X,V,V/np.sqrt(3)

@njit(nogil=True)
def test3(N=10_000):
    '''Accept-reject method for f(x,v,t=0)= 1/sqrt(2*pi)*v^2*e^{-v^2/2}*(1+cos(2*pi*(x+1/2)))'''
    c = 2
    q = lambda x,v:1/np.sqrt(2*np.pi)*np.exp(-v**2/2)
    f = lambda x,v:1/np.sqrt(2*np.pi)*np.exp(-v**2/2)*(1+np.cos(2*np.pi*(x+0.5)))
    #Function to generate random numbers from q
    def rq(n):
        x = np.random.uniform(0,1,size=n)
        v = np.random.normal(0,1,size=n)
        return x,v
    I = np.ones(N)
    n = np.sum(I,dtype=np.int64)
    X = np.zeros(N); V = np.zeros(N)
    while n>0:
        x,v = rq(n)
        #Uniform number to test whether to accept or reject
        U = np.random.uniform(0,1,size=n)
        #Test if reject or accept
        I_n = U <= f(x,v)/(c*q(x,v))
        index_accept = np.where(I)[0][I_n]
        X[index_accept] = x[I_n];V[index_accept] = v[I_n]
        I[index_accept] = 0
        n = np.sum(I,dtype=np.int64)
    return X,V,V

def plot_exact():
    x_axis = np.arange(0,1,1e-2)
    v_axis = np.arange(-4,4,8e-2)
    f = lambda x,v:1/np.sqrt(2*np.pi)*np.exp(-v**2/2)*v**2*(1+np.cos(2*np.pi*(x+0.5))) #test2
    # f = lambda x,v:1/np.sqrt(2*np.pi)*np.exp(-v**2/2)*(1+np.cos(2*np.pi*(x+0.5))) # test3
    X, V = np.meshgrid(x_axis,v_axis)
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of f(x,v,t=0)')
    surf = ax.plot_surface(X, V, f(X,V), rstride=1, cstride=1, edgecolor='none')



if __name__ == '__main__':
    # X = test1(N=30_000)
    # rho = lambda x: 1+np.cos(2*np.pi*(x+0.5))
    # dist = pd.DataFrame(data={'x':X,'q(x)':['Estimation of p(x)' for _ in range(len(X))]})
    # sns.kdeplot(data=dist, x="x",hue='q(x)',linestyle='dashed')
    # plt.plot(np.arange(0,1,0.001),rho(np.arange(0,1,0.001)),label='p(x)')
    # plt.legend(labels=['Estimated p(x)','p(x)'])
    X,V,_ = test2(500_000)
    plot_test(np.vstack((X,V)))
    plot_exact()


    plt.show()
