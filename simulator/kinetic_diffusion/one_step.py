import numpy as np
from typing import Callable,Tuple
from numba import jit_module


#Kinetic one-step operator
def __psi_k(tau,x,v,t):
    return x+v*tau,t+tau


#Diffusive coefficient
def __D(x,v,e,theta,mu,sigma,R):
    return 2*sigma(x)**2/(R(x)**2)*(2*(e-1)+R(x)*theta*(e+1))+(v-mu(x))**2/(R(x)**2)*(1-2*R(x)*theta*e-e**2)

#The KD-operator. Moves the particle(s) according to the kinetic-diffusion algorithm by a distance of dt

def __psi_d(xp,t,v_next,theta,xi,mu,sigma,R):
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
    x_pp = xp + mu(xp)*theta/2 #Intermediate point to deal with heterogenity
    v = v_next.copy() #Velocity of kinetic phase in next step. Drawn at the position of the collision
    e = np.exp(-R(x_pp)*theta)
    dVdx_r = 0#derivative(__D,x_pp,dx=1e-6,args = (v,e,theta,mu,sigma,R))
    Edx = mu(x_pp)*theta + (v-mu(x_pp))*1/R(x_pp)*(1-e) + 0.5*dVdx_r #Equation 31
    Vdx = __D(x_pp,v,e,theta,mu,sigma,R)
    Vdx = np.maximum(Vdx,0) #To avoid numerical errors
    x = xp + Edx + np.sqrt(Vdx)*xi #Update postion by diffusive motion
    t = t + theta
    return x,v,t


#Kinetic-diffusion opertor
def phi_KD(dt,x0,v0,t,tau,xi,mu,sigma,M,R,v_rv=None):
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
    theta = dt - np.mod(tau,dt)
    '''When correlating paths the r.v. for the next step is given in the Coarse
        step by using the ones from the fine step. In that case v_rv is given'''
    v_next = mu(xp)+sigma(xp)*M(xp) if v_rv ==None else  mu(xp)+sigma(xp)*v_rv
    x,v,t = __psi_d(xp,t,v_next,theta,xi,mu,sigma,R)
    return x,v,t,v_next

jit_module(nopython=True,nogil=True, parallel = True)

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
