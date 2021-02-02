import numpy as np
from typing import Callable,Tuple
# from numba.experimental import jitclass



# @jitclass(spec)
class Plasma(object):
    """Class to play role as plasma background."""

    def __init__(self,mu:Callable[[np.ndarray],float],sigma:Callable[[np.ndarray],float],maxwellian = 'default',init_dist = 'default', epsilon = 1):
        '''
        mu: mean of the post-collisional velocity distribution
        sigma: standard deviation of the post-collisional velocity distribution
        Maxwellian: string giving the family of the post-collisional distribtuion. Deafult is normal
        init_dist: type of initial distribution
        epsilon: diffusive parameter. Only used for examples where the diffusive paramter is explicit
        '''
        self.sigma = sigma
        self.mu = mu
        self.maxwellian = maxwellian
        self.init_dist = init_dist
        self.set_background('default')
        self.epsilon = epsilon
    #The post-collisional velocity distribution. By default it is normally distributed

    #@njit
    def M(self,x):
        if self.maxwellian == 'default':
            return np.random.normal(size=x.size)
        else:
            sys.exit("Not a valid post-collisional velocity distribution")

    #Inintial distribution of position and velocity
    #@njit
    def Q(self,N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.init_dist =='default':
            np.random.seed(4208)
            U = np.random.uniform(size=N)
            I = np.random.uniform(size=N)>0.5
            index = np.argwhere(I).flatten()
            x = np.ones(N)
            v = np.zeros(N)
            v_norm = np.append(np.random.normal(size = len(index)),np.random.normal(size = N-len(index)))
            # v_norm = np.append(norm.ppf(U[index]),norm.ppf(U[list(set(range(N))-set(index))]))
            v[index] = (v_norm[0:len(index)] + 10)/self.epsilon#norm.ppf(U[index],loc=10)
            v[list(set(range(N))-set(index))] = (v_norm[len(index):]-10)/self.epsilon#norm.ppf(U[list(set(range(N))-set(index))],loc=-10)
        elif self.init_dist == 'Standard Normal':
            np.random.seed(42)
            x = np.ones(N)
            v_norm = np.random.normal(size=N)
            v = self.mu(x) + self.sigma(x)*v_norm
        else:
            sys.exit('Not valid initial distribution!')
        return x,v,v_norm

    #sets the collision rate
    #@njit
    def set_background(self,type,a=5,b=100):
        self.type = type
        if type == 'B1':
            self.R = lambda x: -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)
            self.dr_x = lambda x: -b*a*(x<=1) + b*a*np.logical_not(x<=1)
            self.a = a
            self.b = b
            self.get_slope = lambda x: -b*a*(x<=1) + b*a*(x>1)
            self.get_intercept = lambda x: (a+1)*b*(x<=1) + (1-a)*b*(x>1)
        elif type == 'B2':
            self.R = lambda x: b*(x <= 1)+ b*(a*(x-1)+1)*(x>1)
            self.dr_x = b*a*(x>1)

        elif type == 'default':
            self.R = lambda x: 1/(self.epsilon**2)
            self.dr_x = lambda x: 0
        else:
            sys.exit('Not a valid type of background')

    #Simpler version of SampleCollision
    #@njit
    def SC(self,x,v,e,test=False):
        if self.type == 'default':
            dtau = 1/self.R(x)*e
        elif self.type == 'B1' or self.type == 'B2':
            if self.a != 0:
                '''The background is piece wise linear and it is necessary to identify
                    particles that might move accross different pieces, e.g. if
                    x>1 and v<0 then the particle might move below 1 where the collision
                    rate is different'''
                n = x.size
                boundaries = np.array([-np.inf,1,np.inf]) #Bins: (-inf,1] and (1,inf]
                num_of_bins = np.size(boundaries)-1
                index = np.arange(0,n,dtype=int)
                dtau = np.zeros(n)
                bins = np.array([np.where(p <= boundaries)[0][0] for p in x],dtype=int)
                direction = np.sign(v).astype(int)
                '''The lower bound of the integral changes. First it is x, but in
                    succesive iterations it is the lower bound of the current bin'''
                lb = x.copy()
                '''Need to subtract previous integrals from exponential number
                    in each iteration and solve for the remainder'''
                e_remainder = e.copy() #The remainder of e after crossing into new domain
                '''Need to update position of particle after crossing into new domain'''
                x_new = x.copy()
                while len(index)>0:
                    # print(f'e_r = {e_remainder[2]}')
                    '''Determine which bin each particle belongs to, given by index of upper bound'''
                    '''Calculate integral within current bin'''
                    I = np.array([quad(lambda x: self.get_slope(xk)*x + self.get_intercept(xk),xk,boundaries[b-d])[0]/vk if (b-d>0 and b-d<num_of_bins) else np.inf for xk,vk,b,d in zip(lb,v[index],bins,(direction==-1))])
                    index_new_domain = np.argwhere(I <= e_remainder[index]).flatten() #Index for particles that cross into different background
                    # print(f'index_new_domain = {index_new_domain}, \n index = {index}')
                    index_new = index[index_new_domain] #In terms of original number of particles
                    # print(f'index_new = {index_new}')
                    index_same_domain = np.argwhere(I > e_remainder[index]).flatten()
                    alpha = self.get_slope(boundaries[bins[index_same_domain]]); beta = self.get_intercept(boundaries[bins[index_same_domain]])
                    # print(f'x1 = {x[1]}, x4 = {x[4]} \n 1 = {boundaries[bins[index_same_domain]][1]}, 2 = {boundaries[bins[index_same_domain]][2]}')
                    # print(f'alpha1 = {alpha[1]}, alpha4 = {alpha[4]}')
                    index_same = index[index_same_domain]
                    dtau[index_same] = dtau[index_same] + (-alpha*x_new[index_same]-beta + np.sqrt((alpha*x_new[index_same]+beta)**2+2*alpha*v[index_same]*e_remainder[index_same]))/(alpha*v[index_same])
                    dtau[index_new] = dtau[index_new] + (boundaries[bins[index_new_domain]-(direction[index_new_domain]==-1)]-x[index_new])/v[index_new]
                    index = index_new.copy()
                    direction = direction[index_new_domain]
                    bins = bins[index_new_domain] + direction
                    lb = boundaries[bins -1]
                    e_remainder[index_new] = e_remainder[index_new] - I[index_new_domain]
                    x_new[index_new] = boundaries[bins - (direction>0)]
                    if test:
                        print(f'dtau = {dtau}')

                    # print(index)
                '''Check if integrals match'''
                if test:
                    print(f'Integral: {[1/v[i]*quad(self.R,x[i],x[i]+v[i]*dtau[i])[0] if 1/v[i]*quad(self.R,x[i],x[i]+v[i]*dtau[i])[0]-e[i]>1e-7 else 0 for i in range(n)]} \n e = {e}')
                    print(f'x0 = {x[0]}, x2 = {x[2]}, x4 = {x[4]}')
                    print(f'v0 = {v[0]}, v2 = {v[2]}, v4 = {v[4]}')
                    sys.exit()
            else:
                dtau = e/self.b
            if np.min(dtau)<0:
                index = np.argwhere(dtau<0).flatten()
                i = index[0]
                print(f'right_cross= {right_cross[0]}')
                print(f'first point in right: {x[right[0]]} and velocity: {v[right[0]]}, index= {right[0]}')
                print(f'i= {i}')
                print(f'x= {x[i]}, v= {v[i]}, e= {e[i]}, ve = {v[i]*e[i]}')
                print(f'alpha= {alpha[i]}, beta = {beta[i]}')
                print(f'index: {index}')
                print(f'integral: {quad(lambda x: alpha[i]*x+beta[i],x[i],x[i]+v[i]*dtau[i])}')
                print(f'integral: lower= {x[i]}, upper= {x[i]+v[i]*dtau[i]}')
                print(f'dtau_plus= {dtau_plus}')
                print(f'dtau_minus= {(-alpha*x-beta - np.sqrt((alpha*x+beta)**2+2*alpha*v*e))/(alpha*v)}')
                sys.exit('Negative collision times!!')
        else:
            sys.exit('No valid background for drawing collision times. Note that default is a constant background. Change by setting variable BG_type')
        return dtau
