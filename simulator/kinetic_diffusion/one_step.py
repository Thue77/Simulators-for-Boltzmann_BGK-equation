import numpy as np
from typing import Callable,Tuple
from numba import jit_module


#Kinetic one-step operator
def __psi_k(tau,x,v,t):
    return x+v*tau,t+tau


#Diffusive coefficient
def __D(x,v,e,theta,mu,sigma,R):
    return sigma(x)**2/(theta*R(x)**2)*(2*(e-1)+R(x)*theta*(e+1))+(v-mu(x))**2/(2*theta*R(x)**2)*(1-2*R(x)*theta*e-e**2)

#Derivative of diffusion coefficient w.r.t. R
def dDdR(x,v,e,theta,mu,sigma,R):
    #Derivative of first term:
    D1 = (sigma(x)**2/(R(x)**2)*(1-e-R(x)*theta*e)  -2*sigma(x)**2/(theta*R(x)**3)*(R(x)*theta+2*e-2+R(x)*theta*e))
    D2 = (v-mu(x))**2*(1/(2*R(x)**2)*(2*e**2-2*e+2*R(x)*theta*e)-1/(theta*R(x)**3)*(1-e**2-2*R(x)*theta*e))
    return D1+D2

#The KD-operator. Moves the particle(s) according to the kinetic-diffusion algorithm by a distance of dt
def __psi_d(xp,t,v_next,theta,z,mu,sigma,R,dR=None,boundary=None):
    '''
    dt: the time step
    v0: initial velocity for kinetc part of the step
    v_next: the velocity in the next kinetic step. The diffusive step must have same average velocity as v_next
    tau: time to next collision
    epsilon: exponential number for calculation of time to first collision in next step
    Notation is adapted to multilevel article page 5 where no diffusive parameter is explicitly included.
    '''
    n = len(xp)
    # dt = np.ones(n)*dt
    # theta = dt - np.mod(tau,dt)
    if dR is None:
        dR_zero = False
    else:
        dR_zero = np.min(dR(xp))==0 and np.max(dR(xp))==0
    x_pp = xp if dR_zero else xp + mu(xp)*theta/2 #Intermediate point to deal with heterogenity
    if boundary is not None: x_pp = boundary(x_pp)
    v = v_next.copy() #Velocity of kinetic phase in next step. Drawn at the position of the collision
    e = np.exp(-R(x_pp)*theta)
    dDdR_dRdx = np.zeros(n) if dR_zero else dDdR(x_pp,v,e,theta,mu,sigma,R)*dR(x_pp)
    A = mu(x_pp) + (v-mu(x_pp))*1/(theta*R(x_pp))*(1-e) + dDdR_dRdx #Equation 31
    D = __D(x_pp,v,e,theta,mu,sigma,R)
    D = np.maximum(D,0) #To avoid numerical errors
    x = xp + A*theta + np.sqrt(2*D*theta)*z #Update postion by diffusive motion
    t = t + theta
    return x,v,t


#Kinetic-diffusion opertor
def phi_KD(dt,x0,v0,t,tau,z,mu,sigma,M,R,v_rv=None,dR=None,boundary=None):
    '''
    dt: step size
    x0: initial positions
    v0: initial velocities
    t: current time for each particle.
    mu: mean of post-collisional distribution
    sigma: standard deviation of post-collisional distribution
    M: post-collisional distribution
    R: collision rate
    '''
    xp,t = __psi_k(tau,x0,v0,t)
    if boundary is not None:
        xp = boundary(xp)
    theta = dt - np.mod(tau,dt)
    '''When correlating paths the r.v. for the next step is given in the Coarse
        step by using the ones from the fine step. In that case v_rv is given'''
    if v_rv is None:
        v_next,v_norm = M(xp)
    else:
        v_norm = v_rv
        v_next = mu(xp)+sigma(xp)* v_norm
    x,v,t = __psi_d(xp,t,v_next,theta,z,mu,sigma,R,dR=dR,boundary=boundary)
    if boundary is not None:
        x = boundary(x)
    return x,v,t,v_norm

jit_module(nopython=True,nogil=True)

#For testing purposes
if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    n = 10
    x0 = np.zeros(n)
    v0 = np.random.normal(size=n)
    dt = 1
    dtau = np.ones(n)*0.1
    mu = lambda x: 0; sigma = lambda x: 1; M = lambda x: np.random.normal(mu(x),sigma(x),size=x.size); SC = lambda n,x,v: np.random.exponential(scale=1,size=n)
    x,v,_ = __psi_k(dt,v0,x0,dtau,mu,sigma,M,SC)
    sns.histplot(x,label=f'After one step of size {dt}')
    plt.show()
