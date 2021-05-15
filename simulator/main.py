from typing import Callable,Tuple
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.stats import wasserstein_distance
from numba import jit,njit,jit_module,prange
import matplotlib.pyplot as plt
import sys
import os
from accept_reject import test1,test2,test3
from splitting.ml import ml_test as APML_test
from kinetic_diffusion.ml import ml_test as KDML_test
from splitting.mc import mc_density_test as APSMC_density_test
from kinetic_diffusion.mc import mc_density_test as KDMC_density_test
from kinetic_diffusion.mc import mc2_par as KMC_par
from kinetic_diffusion.mc import mc1_par as KDMC_par
from splitting.mc import mc2_par as SMC_par
from splitting.mc import mc1_par as APSMC_par

np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument('epsilon', help="value for mean free path",type=float)
parser.add_argument('a', help="parameter in collision rate. Its function depends of the example",type=float)
parser.add_argument('b', help="parameter in collision rate. Its function depends of the example",type=float)
parser.add_argument('-de','--density_est',action='store_true',help='Compares density estimation error. For final example in Thesis')
parser.add_argument('--ml_test_KD',action='store_true',help='Runs Kinetic-Diffusion version of ml_test for final example in Thesis')
parser.add_argument('--ml_test_APS',action='store_true',help='Runs asymptotic preserving splitting(APS) version of ml_test for final example in Thesis')
parser.add_argument('-rt','--radiative_transport',action='store_true',help='Radiative tranport example for MC-methods')
parser.add_argument('-gt','--goldstein_taylor',action='store_true',help='Runs ml_test for APS with Goldstein-Taylor distribution as post-collisional distribution and r(x)=1')
parser.add_argument('-ls','--level_selection',action='store_true',help='Runs level-selction example in KDML')
parser.add_argument('--no_file',action='store_true',help='indicates that no files should be saved when running examples')
parser.add_argument('--paths',help='number of paths to use for any method with a fixed number of paths', type=int)
parser.add_argument('--folder',help='give name of folder to save files in the folder does not need to exist')


args = parser.parse_args()

epsilon = args.epsilon
a = args.a
b = args.b
N = args.paths

density_est = args.density_est; ml_test_KD = args.ml_test_KD; ml_test_APS = args.ml_test_APS
radiative_transport = args.radiative_transport; goldstein_taylor = args.goldstein_taylor
level_selection = args.level_selection;

if not args.density_est and not args.ml_test_KD and not args.ml_test_APS and not args.goldstein_taylor and not args.radiative_transport and not args.level_selection:
    sys.exit('ERROR: no valid example given. Run "main.py -h" for more help')


#Inintial distribution of position and velocity
def Q(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Initial distribution for (x,v)'''
    if level_selection:
        x = np.ones(N)
        # print('Her')
        v_norm = np.random.normal(0,1,size=N)
        v = mu(x) + sigma(x)*v_norm
    elif radiative_transport:
        x = test1(N); v = np.random.uniform(-1,1,size=N)/epsilon
        # x = np.zeros(N); v = np.zeros(N)
        # U = np.random.uniform(size = int(N/3))
        # x[:int(2*N/3)] = np.random.uniform(-0.5,0.5,size=int(2*N/3))
        # x[int(2*N/3):] = (U<=0.5)* np.random.uniform(-1.0,-0.5,size=int(N/3)) + (U>0.5)* np.random.uniform(0.5,1.0,size=int(N/3))
        # U = np.random.uniform(size = int(N/3))
        # v[:int(2*N/3)] = np.random.uniform(-0.75,0.25,size=int(2*N/3))
        # # print((U<=0.5)*np.random.uniform(-1.0,-0.75,size=int(N/3)))
        # v[int(2*N/3):] = (U<=0.25)*np.random.uniform(-1.0,-0.75,size=int(N/3)) + (U>0.25)* np.random.uniform(0.25,1.0,size=int(N/3))
        v_norm = v/sigma(x)
    elif ml_test_KD or ml_test_APS or density_est:
        x,v,v_norm = test2(N)
        v = v/epsilon
    return x,v,v_norm

def Q_nu(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Initial distribution for (x,nu)'''
    if radiative_transport:
        x = test1(N); v = np.random.uniform(-1,1,size=N)
        v_norm = v.copy()
    elif goldstein_taylor:
        U = np.random.uniform(0,1,size=N)
        v =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)
        x = np.zeros(N)
        v_norm = v.copy()
    elif ml_test_KD or ml_test_APS or density_est:
        x,v,v_norm = test2(N)
    return x,v,v_norm

#sets the collision rate
def R(x):
    if radiative_transport:
        return 1/(epsilon**2)
    elif level_selection:
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)
    elif ml_test_KD or ml_test_APS or density_est:
        return 1/epsilon**2*((-a*(x-0.5)+b)*(x<=0.5) + (a*(x-0.5)+b)*(x>0.5))

def r(x):
    if ml_test_KD or ml_test_APS or density_est:
        return (-a*(x-0.5)+b)*(x<=0.5) + (a*(x-0.5)+b)*(x>0.5)
        #return a*x+b
    else:
        return np.ones(len(x))

def dR(x):
    if radiative_transport:
        return 0
    elif level_selection:
        return (x<=1)*(-b*a) + (x>1)*(b*a)
    elif ml_test_KD or ml_test_APS or density_est:
        if a==0:
            return 0
        else:
            return -a/epsilon**2*(x<=0.5) + a/epsilon**2*(x>0.5)

#Anti derivative of R
def R_anti(x):
    if radiative_transport:
        return x/(epsilon**2)
    elif level_selection:
        return (-b*a/2*x**2 + (a+1)*b*x)*(x<=1) + (b*a/2*x**2+(1-a)*b*x)*(x>1)
    elif ml_test_KD or ml_test_APS or density_est:
        return 1/epsilon**2*((-a*(x**2/2-0.5*x)+b*x)*(x<=0.5) + ((-a*(0.5**2/2-0.5*0.5)+b*0.5) + (a*(x**2/2-0.5*x)+b*x) - (a*(0.5**2/2-0.5*0.5)+b*0.5))*(x>0.5))

#Sample Collision
# @njit(nogil=True,parallel = True)
def SC(x,v,e):
    if radiative_transport or a==0:
        dtau = 1/R(x)*e
    elif level_selection:# or density_est or ml_test_APS or ml_test_KD:
        '''The background is piece wise linear and it is necessary to identify
            particles that might move accross different pieces, e.g. if
            x>1 and v<0 then the particle might move below 1 where the collision
            rate is different'''
        n = len(x)
        if level_selection:
            boundaries = np.array([-math.inf,1,math.inf],np.float64) #Bins: (-inf,1] and (1,inf]
        else:
            boundaries = np.array([-math.inf,0.5,math.inf],np.float64)
        num_of_bins = boundaries.size-1
        index = np.arange(0,n,dtype=np.int64)
        dtau = np.zeros(n)
        #The bin is given by the index of the lower boundary
        '''Indicates which domain each particle belongs to. It is given by the
            index of the last boundary that is smalle than x'''
        bins = np.array([np.where(p < boundaries)[0][0] for p in x],dtype=np.int64)-1
        direction = (v>0).astype(np.int64)-(v<0).astype(np.int64)#np.sign(v).astype(np.int64)
        if level_selection:
            slopes = np.array([-b*a,b*a],dtype=np.float64)
            intercepts = np.array([(a+1)*b,(1-a)*b],dtype=np.float64)
        else:
            slopes = 1/epsilon**2*np.array([-a,a],dtype=np.float64)
            intercepts = 1/epsilon**2*np.array([b-a/2,b+a/2],dtype=np.float64)
        '''Need to subtract previous integrals from exponential number
            in each iteration and solve for the remainder. Note the multiplication
            by v. This is because the integration is done w.r.t. x an not t.'''
        e_remainder = np.abs(e*v) #The remainder of e after crossing into new domain
        e_local = e.copy()
        '''Need to update position of particle after crossing into new domain'''
        x_new = x.copy()
        while len(index)>0:
            '''Determine which bin each particle belongs to, given by index of upper bound'''
            '''Calculate integral within current bin'''
            I = integral_to_boundary(x_new[index],bins[index],direction[index],slopes,intercepts)
            index_new_domain = np.argwhere(I <= e_remainder[index]).flatten() #Index for particles that cross into different background
            index_new = index[index_new_domain] #In terms of original number of particles
            index_same_domain = np.argwhere(I > e_remainder[index]).flatten()
            index_same = index[index_same_domain]
            alpha = slopes[bins[index_same]];beta = intercepts[bins[index_same]]
            dtau[index_same] = dtau[index_same] + (-alpha*x_new[index_same]-beta + np.sqrt((alpha*x_new[index_same]+beta)**2+2*alpha*v[index_same]*e_local[index_same]))/(alpha*v[index_same])
            '''If the particle crosses into a new domain, the time it takes to cross
                needs to be added to the tau calculated in the new domain'''
            if level_selection:
                dtau[index_new] = dtau[index_new] + (1-x[index_new])/v[index_new]
            else:
                dtau[index_new] = dtau[index_new] + (0.5-x[index_new])/v[index_new]
            index = index_new.copy()
            # direction = direction[index_new_domain]
            bins[index_new] = bins[index_new] + direction[index_new]
            #location becomes the bound of the new domain
            '''Need to subtract the integral in the old domain from the exponential
                number'''
            e_remainder[index_new] = e_remainder[index_new] - I[index_new_domain]
            e_local[index_new] = e_local[index_new]-I[index_new_domain]/np.abs(v[index_new])
            '''Update x to equal the value of the boundary that it is crossing'''
            x_new[index_new] = boundaries[bins[index_new] + (direction[index_new]<0)]
    elif ml_test_KD or ml_test_APS or density_est:
        if a==0:
            dtau = 1/R(x)*e
        else:
            # '''The background is piece wise linear and it is necessary to identify
            #     particles that might move accross different pieces, e.g. if
            #     x>1 and v<0 then the particle might move below 1 where the collision
            #     rate is different'''
            # n = len(x)
            # boundaries = np.array([-math.inf,0.5,math.inf],np.float64) #Bins: (-inf,1] and (1,inf]
            # num_of_bins = boundaries.size-1
            # index = np.arange(0,n,dtype=np.int64)
            # dtau = np.zeros(n)
            # #The bin is given by the index of the lower boundary
            # '''Indicates which domain each particle belongs to. It is given by the
            #     index of the last boundary that is smalle than x'''
            # bins = np.array([np.where(p < boundaries)[0][0] for p in x],dtype=np.int64)-1
            # direction = (v>0).astype(np.int64)-(v<0).astype(np.int64)#np.sign(v).astype(np.int64)
            # slopes = 1/epsilon**2*np.array([-a,a],dtype=np.float64)
            # intercepts = 1/epsilon**2*np.array([b-a/2,b+a/2],dtype=np.float64)
            # '''Need to subtract previous integrals from exponential number
            #     in each iteration and solve for the remainder. Note the multiplication
            #     by v. This is because the integration is done w.r.t. x an not t.'''
            # e_remainder = np.abs(e*v) #The remainder of e after crossing into new domain
            # e_local = e.copy()
            # '''Need to update position of particle after crossing into new domain'''
            # x_new = x.copy()
            # while len(index)>0:
            #     '''Determine which bin each particle belongs to, given by index of upper bound'''
            #     '''Calculate integral within current bin'''
            #     I = integral_to_boundary(x_new[index],bins[index],direction[index],slopes,intercepts)
            #     index_new_domain = np.argwhere(I <= e_remainder[index]).flatten() #Index for particles that cross into different background
            #     index_new = index[index_new_domain] #In terms of original number of particles
            #     index_same_domain = np.argwhere(I > e_remainder[index]).flatten()
            #     index_same = index[index_same_domain]
            #     alpha = slopes[bins[index_same]];beta = intercepts[bins[index_same]]
            #     dtau[index_same] = dtau[index_same] + (-alpha*x_new[index_same]-beta + np.sqrt((alpha*x_new[index_same]+beta)**2+2*alpha*v[index_same]*e_local[index_same]))/(alpha*v[index_same])
            #     '''If the particle crosses into a new domain, the time it takes to cross
            #         needs to be added to the tau calculated in the new domain'''
            #     dtau[index_new] = dtau[index_new] + (0.5-x[index_new])/v[index_new]
            #     index = index_new.copy()
            #     # direction = direction[index_new_domain]
            #     bins[index_new] = bins[index_new] + direction[index_new]
            #     #location becomes the bound of the new domain
            #     '''Need to subtract the integral in the old domain from the exponential
            #         number'''
            #     e_remainder[index_new] = e_remainder[index_new] - I[index_new_domain]
            #     e_local[index_new] = e_local[index_new]-I[index_new_domain]/np.abs(v[index_new])
            #     '''Update x to equal the value of the boundary that it is crossing'''
            #     x_new[index_new] = boundaries[bins[index_new] + (direction[index_new]<0)]
            # xL = 1;
            x_new = x.copy()
            dtau = np.zeros_like(x);
            #integral value to add to remainder of rhs
            I = np.zeros_like(x);
            #remainder of rhs. Changes when particle moves across boundary
            # e_remainder = np.abs(e*v)
            e_local = e.copy()
            #active paths. Meaning they still need to determine tau
            A = np.ones_like(x).astype(np.bool_)
            #Value at relevant boundary
            x_b = 0.5
            while np.sum(A)>0:
                #slope and intercept of R(x)
                # alpha = a/epsilon**2*(x_new>x_b)-a/epsilon**2*(x_new<=x_b)
                # beta = (x_new<=x_b)*(b-a/2)/epsilon**2 + (x_new>x_b)*(b+a/2)/epsilon**2
                #check if particle is moving in direction of boundary
                I_b = (x_new>x_b)*(v<0)+(x_new<=x_b)*(v>0)
                if np.sum(I_b)>0:
                    # print(I_b)
                    I = np.abs(R_anti(x_b)-R_anti(x_new[I_b]))
                    # print('Integral:')
                    # print(I)
                    # print('x:')
                    # print(x_new[I_b])
                    i_b = np.where(I<np.abs(e[I_b]*v[I_b]))[0]
                    # print('i_b')
                    # print(i_b)
                    index = np.where(I_b)[0][i_b] #np.isin(np.where(I_b)[0],i_b)
                    # print(index)
                    dtau[index] = dtau[index] + (x_b-x_new[index])/v[index]
                    # print('dtau')
                    # print(dtau)
                    temp = np.zeros_like(A); temp[index] =True# np.put(temp,index,1)
                    # print('temp')
                    # print(temp)
                else:
                    temp = np.zeros_like(A)
                I_s = np.logical_not(temp)*A
                # print('I_s')
                # print(I_s)
                # e_remainder[I_b[i_b]] = e_remainder[I_b[i_b]] - I
                # print(alpha)
                # print(v)
                # print(I_s)
                r = roots(x_new[I_s],v[I_s],e_local[I_s])
                # print('root')
                # print(r)
                dtau[I_s] = dtau[I_s] + r#(-alpha[I_s]*x_new[I_s]-beta[I_s] + np.sqrt((alpha[I_s]*x_new[I_s]+beta[I_s])**2+2*alpha[I_s]*v[I_s]*e_local[I_s]))/(alpha[I_s]*v[I_s])
                if np.sum(I_b)>0:
                    x_new[temp] = (x_new[temp]>x_b)*0.5+(x_new[temp]<=x_b)*(0.5+1e-7)
                    # print(x_new)
                    e_local[temp] = e_local[temp]-I[i_b]/np.abs(v[temp])
                    # print('e_local')
                    # print(e_local)
                A = I_b*temp

                # #integral to boundary
                # A[I_n] = a/2*x_new[I_n]**2 + b*x_new[I_n]
                # A[I_p] = a/2*(xL**2-x_new[I_p]**2)+b*(x_L-x_new[I_p])
                # I_new_domain = A[I]<e_remainder[I]
                # I_same_domain = A[I]>e_remainder[I]
                # dtau[I_new_domain] = dtau[I_new_domain] + (x_b[I_new_domain]-x_new[I_new_domain])/v[I_new_domain]
                # dtau[I_same_domain] = (-a*x_new[I_same_domain]-b + np.sqrt((a*x_new[I_same_domain]+b)**2+2*a*v[I_same_domain]*epsilon**2*e_local[I_same_domain]))/(a*v[I_same_domain])
                # e_remainder[I_new_domain] = e_remainder[I_new_domain]-A[I_new_domain]
                # e_local = e_local - A/np.abs(v)
                # x_new[I_new_domain] = x_b[I_new_domain]
                # I = I_new_domain.copy()
                # I_p=I_p*I; I_n=I_n*I;
    return dtau

@njit(nogil=True,parallel=True)
def roots(x,v,e):
    if density_est or ml_test_APS or ml_test_KD:
        alpha = -a/epsilon**2*(x<=0.5) + a/epsilon**2*(x>0.5)
        beta = (b+a/2)/epsilon**2*(x<=0.5) + (b-a/2)/epsilon**2*(x>0.5)
        pc = -e*v; pb = alpha*v*x+beta*v; pa = alpha/2*v**2
        p = np.concatenate((pc.reshape((1,e.size)),pb.reshape((1,e.size)),pa.reshape((1,e.size))),axis=0).T
        r = np.zeros_like(x)
        for i in prange(x.size):
            # r[i] = np.polynomial.polynomial.polyroots(p[i,:])
            r[i] = np.roots(np.flip(p[i,:]))[1]
        return r

@njit(nogil=True,parallel = True)
def integral_to_boundary(x,bins,direction,slopes,intercepts):
    '''
    Calculates integral of R to the boundary in the direction of the velocity.
    only works for B1 and B2. Integral is always positive. The affect of the
    sign of v has been removed


    boundaries: numpy array of boundaries for all domains
    slopes: numpy array of slope in each domain
    intercepts: numpy array of intercepts in each domain
    '''
    #Find function value where the domain changes
    B = slopes*1 + intercepts
    #index indicating finite integrals
    index = np.argwhere((bins==1)*(direction<0) + (bins==0)*(direction>0)).flatten()
    I = np.ones(len(x))*math.inf
    if level_selection:
        for j in prange(len(index)):
            i = index[j]
            I[i] = abs(x[i]-1)*B[bins[i]]+(abs(x[i]-1)*(slopes[bins[i]]*x[i]+intercepts[bins[i]]-B[bins[i]]))/2
    else:
        for j in prange(len(index)):
            i = index[j]
            I[i] = abs(x[i]-0.5)*B[bins[i]]+(abs(x[i]-0.5)*(slopes[bins[i]]*x[i]+intercepts[bins[i]]-B[bins[i]]))/2
    return I


def mu(x):
    return 0


def sigma(x):
    if ml_test_KD or ml_test_APS or density_est:
        return 1/epsilon
    elif level_selection:
        return 1
    elif radiative_transport:
        return np.sqrt(1/3)/epsilon

def M(x):
    '''Distribution of velocity scaled by epsilon, i.e. v'''
    if radiative_transport:
        # v_norm = np.random.uniform(0,1)
        # v_next = (1-v_norm*2)/epsilon
        v_next = np.random.uniform(-1,1,size=len(x))/epsilon
        v_norm = v_next/sigma(x)
    elif ml_test_KD or ml_test_APS or density_est:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = v_norm/epsilon
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm
    return v_next,v_norm

def M_nu(x):
    '''Distribution of velocity NOT scaled by epsilon, i.e. nu'''
    if radiative_transport:
        # v_norm = np.random.uniform(0,1)
        # v_next = (1-v_norm*2)/epsilon
        v_next = np.random.uniform(-1,1,size=len(x))
        v_norm = v_next.copy()
    elif goldstein_taylor:
        U = np.random.uniform(0,1,size=len(x))
        v_next =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)
        v_norm = v_next.copy()
    elif ml_test_KD or ml_test_APS or density_est:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = v_norm.copy()
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm
    return v_next,v_norm

def boundary_periodic(x):
    x0 = 0; xL = 1#/epsilon
    l = xL-x0 #Length of x domain
    I_low = (x<x0); I_high = (x>xL);
    x_new = x.copy()
    x_new[I_low] = xL-((x0-x[I_low])%l)#x_new[I_low] + l#
    x_new[I_high] = x0 + ((x[I_high]-xL)%l)#x_new[I_high] - l#
    return x_new

def boundary(x):
    return x

'''Function related to the quantity of interest, E(F(X,V))'''
def F(x,v=0):
    if goldstein_taylor: #or test == 'num_exp_ml':
        return x**2
    else:
        return x


jit_module(nopython=True,nogil=True)


if __name__ == '__main__':
    if ml_test_APS:
        if N is None:
            N = 120_000
        N0=40; T=1; dt_list = T/2**np.arange(0,17,1); E2=np.array([0.01,0.0001,1e-6],dtype=np.float64); t0=0; M_t=2
        if args.no_file:
            logfile=None
        else:
            logfile = open(f'logfile_APS_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
        APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile)
    if ml_test_KD:
        if N is None:
            N = 120_000
        N0=40; T=1; dt_list = T/2**np.arange(0,17,1); E2=np.array([0.01,0.0001,1e-6],dtype=np.float64); t0=0; M_t=2
        if args.no_file:
            logfile=None
        else:
            logfile = open(f'logfile_KD_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
        KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,boundary=boundary)
    if goldstein_taylor:
        if N is None:
            N=120_000;
        N0=40; M_t = 2; t0=0;T=0.5
        dt_list = T/2**np.arange(0,17,1); E2=np.array([0.01,0.0001,1e-6],dtype=np.float64)
        if args.no_file:
            logfile=None
        else:
            logfile = open(f'logfile_APS_Goldstein_Taylor_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
        APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile)
    if density_est:
        if args.folder:
            path = os.getcwd()
            path = path+ '\\'+args.folder
            os.chdir(path)
        if N is None:
            N = 120_000
        if N%8!=0 or N%80!=0:
            sys.exit('Please provide a number of paths divisible by 8 and 80')
        if not args.no_file:
            file = open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
        N0=40; T=1; dt_list = T/2**np.arange(0,17,1); t0=0; M_t=2
        if True:
            # x = np.array([0.8,0.55]); v = np.array([0.9,-0.6]); e = np.array([0.02254558,0.59401347])
            # tau = SC(x,v,e)
            # print(f'tau = {tau}')
            # print(f'intercepts: {(1/epsilon**2*(b-a/2),1/epsilon**2*(b+a/2))}')
            # print(f'slopes: {(a/epsilon**2,-a/epsilon**2)}')
            # print(f'e = {e}')
            print('Testing consistency')
            x_KD=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary)
            dist = pd.DataFrame(data={'x':x_KD,'Method':['KD' for _ in range(N)]})
            print('Done with KMC')
            x_AP = SMC_par((T-t0)/2**19,t0,T,N,epsilon,Q_nu,M_nu,boundary,r)
            dist = dist.append(pd.DataFrame(data={'x':x_AP,'Method':['Splitting' for _ in range(N)]}))
            print(wasserstein_distance(x_AP,x_KD))
            sns.kdeplot(data=dist, x="x",hue='Method',cut=0,common_norm=False)
            plt.show()
            if not args.no_file:
                np.savetxt(f'density_exact_AP_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt',x_AP)
                np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt',x_KD)
        else:
            print(f'{N} paths used to simulate exact density')
            x_std = SMC_par((T-t0)/2**20,t0,T,N,epsilon,Q_nu,M_nu,boundary,r)
            print('Done with exact')
            print(f'{N/10} paths used to estimate density with APSMC and KDMC')
            W,err=APSMC_density_test(dt_list,M_t,t0,T,N/10,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std)
            print(W)
            if not args.no_file:
                np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='Error for APS')
            print('APS is done')
            x_std=KDMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary)
            print('Done with exact')
            W,err = KDMC_density_test(dt_list,Q,t0,T,N/10,mu,sigma,M,R,SC,dR=dR,boundary=boundary,x_std=x_std)
            if not args.no_file:
                np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='Error for KD')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Delta t$')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.show()
    if radiative_transport:
        '''Numerical experiemnt on radiative transport with
        rho(x,0) = 1+cos(2*pi*(x+0.5)) and V ~ U(-1,1)'''
        x_lim = (0,1); v_lim = (-1,1)
        N = 500_000
        dt = 0.1e-5;t0=0;T=0.1
        x = APSMC_par(dt,t0,T,N,epsilon,Q_nu,M_nu,boundary_periodic,r)
        print('Done with APSMC')
        dist = pd.DataFrame(data={'x':x,'Method':['APS' for _ in range(N)]})
        x = KDMC_par(dt,N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary_periodic)
        print('Done with KDMC')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['KD' for _ in range(N)]}))
        x = SMC_par(dt,t0,T,N,epsilon,Q_nu,M_nu,boundary_periodic,r)
        print('Done with SMC')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['SS' for _ in range(N)]}))
        x=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary_periodic)
        print('Done with KMC')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['Kinetic' for _ in range(N)]}))
        sns.kdeplot(data=dist, x="x",hue='Method',cut=0,common_norm=False)
        plt.show()