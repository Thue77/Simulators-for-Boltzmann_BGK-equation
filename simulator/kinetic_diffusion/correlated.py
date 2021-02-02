import numpy as np
from .one_step import phi_KD
from typing import Callable,Tuple
from numba import jit_module

'''Method for correlating paths of different levels in the Multilevel Monte Carlo
    method'''

def correlated(dt_f,dt_c,x0,v0,v_l1_next,t0,T,mu:Callable[[np.ndarray],np.ndarray],sigma:Callable[[np.ndarray],np.ndarray],M:Callable[[np.ndarray,int],np.ndarray],R:Callable[[np.ndarray],np.ndarray],SC:Callable[[int],np.ndarray]):
    '''
    dt_f: time step for fine level
    dt_c: time step for coarse level
    x0: initial position
    v0: initial velocity
    '''
    # np.random.seed(43)
    n = len(x0)
    '''Output fine variables. t_out is not in the output but it is necessary to
       know which x to save in x_out. Need to save the last x s.t. t+tau>T.
       Then if t<T we need to move x_out by T-t'''
    x_out = x0.copy(); t_out = np.zeros(n); v_out = v0.copy()
    e = np.random.exponential(1,size=n)
    tau_k1 = SC(x0,v0,e); tau_k2 = tau_k1.copy()
    tau_out = tau_k1.copy()
    t_k1 = np.ones(n)*t0; t_k2 = np.ones(n)*t0
    v_k1 = v0.copy(); v_k2 = v0.copy(); x_k1 = x0.copy(); x_k2 = x0.copy()
    I1 = (t_k1+tau_k1)<T; I2 = (t_k2+tau_k2)<T
    I12 = np.logical_or(I1,I2)
    I12_temp = I12.copy()
    active = np.where(I12)[0]; n = len(active) #Paths where either fine or coarse is active
    i1 = np.where(I1)[0]; n1 = len(i1) #Active fine particle
    i2 = np.where(I2)[0]; n2 = len(i2) #Active coars particle
    last = i2
    count=0
    while n>0:
        v_save = c_np(np.empty((n2,0)),v_l1_next[last]); e_save = np.empty((n2,0)); xi_save = np.empty((n2,0))
        theta = np.empty((n2,0))
        tau = np.empty((n2,0)); x_save = np.empty((n2,0))
        n_temp = n;
        move = active.copy() #Paths where either fine or coarse is active
        '''Determine which r.v. to save'''
        Srv_n = isin_np(move,i2); save_rv = np.where(Srv_n)[0] #Index w.r.t. n
        Srv_n2 = isin_np(i2,move); save_rv2 = np.where(Srv_n2)[0] #Index w.r.t. n2
        while True:
            xi_rv = np.random.normal(0,1,size=n)
            # v_save = np.c_[v_save,v_rv*Srv]; e_save = np.c_[e_save,e_rv*Srv]; xi_save = np.c_[xi_save,xi_rv*Srv]
            input = np.zeros(n2); input[save_rv2] = xi_rv[save_rv].copy(); xi_save = c_np(xi_save,input)
            theta_k1 = dt_f - np.mod(tau_k1[move],dt_f)
            x_k1[move],v_k1[move],t_k1[move],v_next = phi_KD(dt_f,x_k1[move],v_k1[move],t_k1[move],tau_k1[move],xi_rv,mu,sigma,M,R)# self.S_KD(dt_f,x_k1[move],v_k1[move],t_k1[move],(self.mu(x_k1[move])+self.sigma(x_k1[move])*v_rv),tau_k1[move],e_rv,xi_rv)
            input = np.zeros(n2); input[save_rv2] = v_next[save_rv].copy(); v_save = c_np(v_save,input)

            input = np.zeros(n2); input[save_rv2] = x_k1[move[Srv_n]].copy(); x_save = c_np(x_save,input)
            input = np.zeros(n2); input[save_rv2] = theta_k1[save_rv].copy(); theta = c_np(theta,input)

            '''Find out if any path is beyond the time scope and needs to be
               saved in  the _out variables'''
            done = np.where(((np.ceil((t_k1[move]+tau_k1[move])/dt_f)*dt_f)>T)*(x_out[move]==x0[move]))[0]
            x_out[move[done]] = x_k1[move[done]].copy(); t_out[move[done]] = t_k1[move[done]].copy()
            v_out[move[done]] = v_k1[move[done]].copy(); tau_out[move[done]] = tau_k1[move[done]].copy()

            '''Check which particles still need to move'''
            I_move = ((np.ceil((t_k1+tau_k1)/dt_f)*dt_f) <= (np.ceil((t_k2+tau_k2)/dt_c)*dt_c))*I12
            move = np.where(I_move)[0]
            n = len(move)
            if n==0:
                break
            e_rv = np.random.exponential(1,size=n);
            tau_k1[move] = SC(x_k1[move],v_k1[move],e_rv)
            '''Determine which r.v. to save'''
            Srv_n = isin_np(move,i2)
            Srv_n2 = isin_np(i2,move)
            save_rv = np.where(Srv_n)[0] #Index w.r.t. n
            save_rv2 = np.where(Srv_n2)[0] #Index w.r.t. n2
            input = np.zeros(n2); input[save_rv2] = tau_k1[move[Srv_n]].copy(); tau = c_np(tau,input)
            input = np.zeros(n2); input[save_rv2] = e_rv[save_rv].copy(); e_save = c_np(e_save,input)
        n = n_temp
        if n2 > 0:
            Srv_n = isin_np(active,i2); save_rv = np.where(Srv_n)[0] #Index w.r.t. n
            v_l1_next,col = get_last_nonzero_col(v_save)#v_save[range(n2),(v_save!=0).cumsum(1).argmax(1)]#np.choose((v_save!=0).cumsum(1).argmax(1),v_save.T)#v_k1[range(len(index_k2)),(v_k1!=0).cumsum(1).argmax(1)]#np.choose((v_k1!=0).cumsum(1).argmax(1),v_k1.T) #Take velocities from fine path for active particles(particles for which the next collision happens before T). Must correspond to the last non-zero contribution. Gives velocity in next coarse step
            v_save = set_last_nonzero_col(v_save,col)
            temp = integral_of_R(R,(np.ceil((t_k2[i2]+tau_k2[i2])/dt_c)*dt_c),t_k1[i2],x_k1[i2],mu(x_k1[i2])+sigma(x_k1[i2])*v_l1_next)
            e_old,_ = get_last_nonzero_col(e_save)#e_save[range(n2),(e_save!=0).cumsum(1).argmax(1)]#np.choose((e_save!=0).cumsum(1).argmax(1),e_save.T)
            e_k2 = e_old - temp
            xi_k2 = get_xi(v_save[:,1:-1],x_save,xi_save,tau,theta,sigma,R(x_save))
            x_k2[i2],v_k2[i2],t_k2[i2],_ = phi_KD(dt_c,x_k2[i2],v_k2[i2],t_k2[i2],tau_k2[i2],xi_k2,mu,sigma,M,R,v_rv=v_l1_next)# self.S_KD(dt_c,x_k2[i2],v_k2[i2],t_k2[i2],(self.mu(x_k2[i2])+self.sigma(x_k2[i2])*v_l1_next),tau_k2[i2],e_k2,xi_k2)
        '''Test if any paths are still active'''
        I1 = (t_k1+tau_k1)<T; I2 = (t_k2+tau_k2)<T
        I12 = np.logical_or(I1,I2)
        temp = i2.copy()
        active = np.where(I12)[0]; n = len(active) #Paths where either fine or coarse is active
        if n==0:
            break
        i2 = np.where(I2)[0]; n2 = len(i2) #Active coarse particle
        '''Find indexes to determine which of v_l1_next to use in next step'''
        last = np.where(isin_np(temp,i2))[0]
        count+=1
    I = t_out<T
    if np.sum(I)>0:
        index = np.argwhere(I).flatten()
        x_out[index] = x_out[index] + v_out[index]*(T-t_out[index])
        t_out[index] = T
    I = t_k2<T
    if np.sum(I)>0:
        index = np.argwhere(I).flatten()
        x_k2[index] = x_k2[index] + v_k2[index]*(T-t_k2[index])
    return x_out,x_k2


def get_xi(v,x,xi,tau,theta,sigma,R):
    '''
    v: Generated normal numbers for kinetic phases in fine steps not including the first
    xi: Generated normal numbers for diffusive phases in fine steps
    tau: Times to collisions for fine steps not including the first time
    theta: Lengths of diffusive phases in fine steps
    '''
    zeta = 1/R*(1-np.exp(-R*theta))*(theta!=0)
    tau_p = tau + zeta[:,:-1]
    theta_p = theta - zeta
    sigma_v = sigma(x)
    alpha = tau_p*sigma_v*(v!=0)
    beta = np.sqrt(2*sigma_v**2/R**2*(np.exp(-R*theta_p)+R*theta_p-1))
    num = (beta[:,0]*xi[:,0]+np.sum(alpha*v+beta[:,1:]*xi[:,1:],axis=1))
    denom = (np.sqrt(beta[:,0]**2+np.sum(alpha**2+beta[:,1:]**2,axis=1)))
    I = np.argwhere(denom > 0).flatten()
    temp = np.zeros(v.shape[0])
    temp[I] = num[I]/denom[I]
    return temp

#Needs to be adapted heterogeneous background. Scipy is not part of numba yet

def integral_of_R(R,t_l1,t_l,x,v):
    '''This function calculates integrals of R from a to b if a<b
    a,b: numpy arrays of start and end times
    '''
    index = np.argwhere(t_l1>t_l).flatten()
    I = np.zeros(len(t_l1),dtype=np.float64)
    for i in index:
        I[i] = R(x[i])*(t_l1[i]-t_l[i])#1/v[i]*quad(self.R,x[i],x[i]+v[i]*(t_l1[i]-t_l[i]))[0]
    return I


def c_np(A,b):
    '''Alternative to np.c_. Needed for numba'''
    return np.vstack((A.T,b.reshape(1,len(b)))).T


def isin_np(A,B):
    '''Alternative ti np.isin. Needed for numba'''
    return np.array([a in B for a in A])

def get_last_nonzero_col(A):
    '''Get last nonzero column from 2d array'''
    I = A!=0
    index_col = np.array([np.sum(a)-1 for a in I])
    return np.array([A[i,index_col[i]] for i in range(A.shape[0])]),index_col


def set_last_nonzero_col(A,index):
    '''Set last nonzero column to zero'''
    B = A.copy()
    for i in range(A.shape[0]):
        B[i,index[i]] = 0
    return B

jit_module(nopython=True,nogil=True)
