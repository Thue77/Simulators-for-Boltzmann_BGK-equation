import numpy as np
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from accept_reject import test1
import seaborn as sns
import pandas as pd
from numba import njit
import sys
'''Solve kinetic equation deterministically using central difference'''
eps = 1#0.05
#Step sizes
dx = 0.1#input('step size for x: ')
p = 10#input('step size for v: ')
dv = 1/p
I_t = (0,0.5)
dt = 0.01

#Domain
start = (-1,-1) #coordinate of lower left corner in descrete domain
end = (1,1)

#Axis
x_axis = np.arange(start[0],end[0],dx)
v_axis =  np.append(-(2*np.flip(np.arange(1,p+1))-1)/(2*p),(2*np.arange(1,p+1)-1)/(2*p))
#Number of increments
N_x = int((end[0]-start[0])/dx)
# N_v = int((end[1]-start[1])/h_v)


'''-----------Deterministic solver of kinetic equation(Not AP, but standard)--------------'''

def initialise_kinetic():
    f = np.zeros((N_x,2*p)) #Vector f_hat
    err = 1e-4
    for i,x in enumerate(x_axis):
        for j,v in enumerate(v_axis):
            in_d = (x<=(0.5+err) and x>=-(0.5+err))*(v>=-(0.75+err) and v<=(0.25+err))
            f[i,j] = 2*dx*dv*in_d + dx*dv*(not in_d)
    return f

'''Instead of advection matrix'''
def Advection_CD():
    '''relevant indexes for advection term in kinetic equation when using central
    difference scheme and periodic boundary conditions'''
    ileft = np.append(np.array([N_x-1]),np.arange(0,N_x-1))
    iright = np.append(np.arange(1,N_x),np.array([0]))
    return ileft,iright

def equilibrium(f):
    n,m = f.shape
    rho = np.sum(f,axis=0)



'''Use Backward for transport step and forward Euler for collision step
 to solve f_t + v/eps*f_x = 1/eps**2(M*rho-f)'''
def kinetic_FE():
    d = 1/3
    nu = 0.1
    dt = nu*dx**2/d
    v = v_axis.copy()
    f = initialise_kinetic()
    ileft,iright = Advection_CD()
    '''Forward Euler discretization of kinetic equation'''
    steps = int((I_t[1]-I_t[0])/dt)
    # A_inv = np.linalg.inv(np.identity((N_x+1)*(N_v+1))+dt*block_multiply(A,v)/eps)
    for i in range(steps):
        f = f - dt/eps*(v*(f[ileft,:]-f[iright,:]))/(2*dx)#np.matmul(A_inv,f) #Transport step
        f = f+ dt/eps**2*(0.5*np.sum(f,axis=0)-f)#(equilibrium(f,N_x,N_v)-f) #Collision step
    rho = np.sum(f,axis=1)
    plt.plot(x_axis,rho)
    plt.show()


'''---------Deterministic solver of heat equation------------'''

def diffusive_CD():
    '''relevant indexes for diffusion term in heat equation when using central
    difference scheme and periodic boundary conditions'''
    ileft = np.append(np.array([N_x-1]),np.arange(0,N_x-1))
    icentral = np.arange(0,N_x)
    iright = np.append(np.arange(1,N_x),np.array([0]))
    return ileft,icentral,iright

def initialise_diffusive():
    '''Initialise rho in heat equation'''
    rho = np.zeros(N_x)
    err = 1e-5 #tolerance for comparison of floats
    for i,x in enumerate(x_axis*eps):
        if x>-(0.5-err)*eps and x<(0.5-err)*eps:
            rho[i] = dx
        elif (x>=-(0.5+err)*eps and x<=-(0.5-err)*eps) or (x<=(0.5+err)*eps and x>=(0.5-err)*eps):
            rho[i] = 3/4*dx
        else:
            rho[i] = dx/2
    return rho

def diffusive_FE():
    '''Solve heat equation with forward Euler time discretization
    This method solves for rho and not rho_{epsilon}. To make sure that
    rho_{epsilon}(x,t=2.5) is obtained, the x_axis must be scaled by 1/epsilon and
    the time must be scaled by 1/epsilon^2'''
    global dx
    d = 1/3
    nu = 0.3
    dx = dx*eps
    dt = nu*dx**2/d
    # print(dt)
    rho = initialise_diffusive()
    # plt.plot(x_axis,rho)
    # plt.show()
    ileft,icentral,iright = diffusive_CD()
    t = I_t[0]; T = I_t[1]*eps**2
    while t<T:
        rho = rho + dt*d/(dx**2)*(rho[iright]-2*rho[icentral]+rho[ileft])
        t+=dt
    rho = np.append(rho,rho[0])
    plt.plot(np.append(x_axis,end[0]),rho)
    plt.show()

'''---------Deterministic solver of altered heat equation with solution given below------------'''
'''Altered equation is: u_t = u_{xx} + b(x,t), x in [-1,1)'''

def heat_altered():
    rho_ref = lambda t,x: np.exp(-t)*np.cos(5*np.pi*x)
    b = lambda t,x: (25*np.pi**2-1)*np.exp(-t)*np.cos(5*np.pi*x)
    rho = rho_ref(0,x_axis)
    ileft,icentral,iright = diffusive_CD()
    t = 0; T = 2
    dt = dx**2/2
    while t<T:
        rho = rho + dt*((rho[ileft]-2*rho[icentral]+rho[iright])/dx**2 + b(t,x_axis))
        t += dt
    plt.plot(x_axis,rho,label='numerical solution')
    plt.plot(x_axis,rho_ref(t,x_axis),label='exact solution')
    plt.ylim(-1,1)
    plt.legend()
    plt.show()





'''----------Simulation of heat equation(asymptotic limit) via MC method with d = <v^2>----------'''
def Q(N):
    x = np.zeros(N); v = np.zeros(N)
    U = np.random.uniform(size = int(N/3))
    x[:int(2*N/3)] = np.random.uniform(-0.5,0.5,size=int(2*N/3))
    x[int(2*N/3):] = (U<=0.5)* np.random.uniform(-1.0,-0.5,size=int(N/3)) + (U>0.5)* np.random.uniform(0.5,1.0,size=int(N/3))
    U = np.random.uniform(size = int(N/3))
    v[:int(2*N/3)] = np.random.uniform(-0.75,0.25,size=int(2*N/3))
    # print((U<=0.5)*np.random.uniform(-1.0,-0.75,size=int(N/3)))
    v[int(2*N/3):] = (U<=0.25)*np.random.uniform(-1.0,-0.75,size=int(N/3)) + (U>0.25)* np.random.uniform(0.25,1.0,size=int(N/3))
    return x,v

@njit(nogil=True)
def boundary(x):
    x0 = start[0]*eps; xL = end[0]*eps
    l = xL-x0 #Length of x domain
    I_low = (x<x0); I_high = (x>=xL);
    x[I_low] = xL-((x0-x[I_low])%l)
    x[I_high] = x0 + ((x[I_high]-xL)%l)
    return x

njit(nogil=True)
def heat_MC():
    # global dx
    # dx = dx
    d = 1/3#np.sum(v**2)/(2*p)
    nu = 0.3
    dt = 1e-3#nu*dx**2/d
    N = 1_500_000
    # X,V = Q(N)
    X = test1(N);V = np.random.uniform(start[1],end[1],size=N)
    X = X*eps
    t = 0
    while t<0.1*eps:
        X = X + np.sqrt(d*dt)*np.random.normal(0,1,size=N)
        X = boundary(X)
        t+=dt
    X = X/eps
    return X



if __name__ == '__main__':
    # diffusive_FE()
    # heat_altered()
    kinetic_FE()
    # X = heat_MC()
    # dist = pd.DataFrame(data={'x':X})
    # sns.kdeplot(data=dist, x="x",cut=0)
    # plt.show()
