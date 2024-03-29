from typing import Callable,Tuple
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.stats import wasserstein_distance
from numba import jit,njit,jit_module
from numba import prange
import matplotlib.pyplot as plt
import sys
import os
import time
from accept_reject import test1,test2,test3
from splitting.ml import ml_test as APML_test
from kinetic_diffusion.ml import ml_test as KDML_test
from splitting.mc import mc_density_test as APSMC_density_test
from kinetic_diffusion.mc import mc_density_test as KDMC_density_test
from kinetic_diffusion.mc import mc2_par as KMC_par
from kinetic_diffusion.mc import mc1_par as KDMC_par
from kinetic_diffusion.mc import KDMC
from splitting.mc import mc as APSMC
from splitting.mc import mc2_par as SMC_par
from kinetic_diffusion.mc import Kinetic as KMC
from splitting.mc import mc_standard as SMC
from splitting.mc import mc1_par as APSMC_par
from splitting.correlated import correlated_test,correlated,correlated_ts
from kinetic_diffusion.correlated import correlated as KDCOR
from splitting.ml import convergence_tests as APS_conv
from kinetic_diffusion.ml import convergence_tests as KD_conv

np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument('epsilon', help="value for mean free path",type=float)
parser.add_argument('a', help="parameter in collision rate. Its function depends of the example",type=float)
parser.add_argument('b', help="parameter in collision rate. Its function depends of the example",type=float)
parser.add_argument('-de','--density_est',action='store_true',help='Compares density estimation error. For final example in Thesis')
parser.add_argument('--ml_test_KD',action='store_true',help='Runs Kinetic-Diffusion version of ml_test for final example in Thesis. Remember to give type of test by specifying --type')
parser.add_argument('--ml_test_APS',action='store_true',help='Runs asymptotic preserving splitting(APS) version of ml_test for final example in Thesis. Remember to give type of test by specifying --type')
parser.add_argument('--ml_cost',action='store_true',help='Plots cost of all ML-methods based on the data in the given folder. Remember to give type of test by specifying --type')
parser.add_argument('--type',help='Type of test to run for the ml_test_KD and ml_test_KD. Choose a convergence test, a complexity test or by giving the resepctive arguments "conv", "comp" or "both".')
parser.add_argument('-rt','--radiative_transport',action='store_true',help='Radiative tranport example for MC-methods')
parser.add_argument('-ct','--correlation_test',action='store_true',help='Plot to compare fine and coarse path for new correlation. Homogenous background is used.')
parser.add_argument('-gt','--goldstein_taylor',action='store_true',help='Runs ml_test for APS with Goldstein-Taylor distribution as post-collisional distribution and r(x)=1')
parser.add_argument('-ls','--level_selection',action='store_true',help='Runs level-selction example in KDML')
parser.add_argument('-os','--one_step_dist',action='store_true',help='plots distribution after one step of KDMC method to compare with Figure 4 in KDMC article by Bert Mortier')
parser.add_argument('-ctt','--correlated_time_test',action='store_true',help='Test the time is takes to correlate paths as a function of the step size. Test is done for all three kinds of correlation. For a comparison with non-numba runs, numba must be deactivated manually in the code. Homogenous background is used')
parser.add_argument('-sf','--save_file',action='store_true',help='indicates that files should be saved when running examples')
parser.add_argument('-uf','--use_file',default=False,action='store_true',help='indidcates if existing files should be used. In case it should be, remember to give the appropriate folder!')
parser.add_argument('--paths',help='number of paths to use for any method with a fixed number of paths', type=int)
parser.add_argument('--folder',help='give name of folder to save files in the folder does not need to exist')
parser.add_argument('-rev','--reverse_splitting',action='store_true',help='Variable for ML-testing with splitting approach. Indicates if reversed splitting should be used.')
parser.add_argument('-std','--standard_correlation',action='store_true',help='Variable for ML-testing with splitting approach. Indicates if standard correlation method should be used.')
parser.add_argument('-diff','--altered_diff_coef',action='store_true',help='Variable for ML-testing with splitting approach. Indicates if altered diffusive coefficient should be used.')
parser.add_argument('-pc','--post_collisional',action='store_true',help='If given, the initial velocity distribution corresponds to the post-collisional distribution')
parser.add_argument('-sep','--separator',type=str,help='File path separator used in paths on your system. Default is for Windows 10')
parser.add_argument('-dl','--diffusion_limit',action='store_true',help='Test diffusion limit of the models')


args = parser.parse_args()

epsilon = args.epsilon
a = args.a
b = args.b
N = args.paths


density_est = args.density_est; ml_test_KD = args.ml_test_KD; ml_test_APS = args.ml_test_APS; ml_cost = args.ml_cost; type = args.type
radiative_transport = args.radiative_transport; goldstein_taylor = args.goldstein_taylor
level_selection = args.level_selection; correlation_test = args.correlation_test; correlated_time_test = args.correlated_time_test
uf = args.use_file
rev = args.reverse_splitting; diff = args.altered_diff_coef; std = args.standard_correlation
post_collisional = args.post_collisional
diffusion_limit = args.diffusion_limit
one_step_dist = args.one_step_dist


if not args.ml_cost and not args.one_step_dist and not args.diffusion_limit and not args.correlated_time_test and not args.correlation_test and not args.density_est and not args.ml_test_KD and not args.ml_test_APS and not args.goldstein_taylor and not args.radiative_transport and not args.level_selection:
    sys.exit('ERROR: no valid example given. Run "main.py -h" for more info')

if std and not rev:
    sys.exit('INPUTERROR: The use of std is only to indicate the use standard correlation in the case of reverse splitting. Please add -rev')

if uf and args.save_file:
    sys.exit('ERROR: Cannot both use existing file and save result')

if args.save_file:
    if args.folder:
        print(f'files are saved in {args.folder}')
    else:
        print('files are saved in the working directory')

#Inintial distribution of position and velocity
def Q(N) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Initial distribution for (x,v)'''
    if one_step_dist:
        U = np.random.uniform(0,1,size=N)
        I = np.random.uniform(0,1,size=N)>0.5
        index = np.argwhere(I).flatten()
        index_not = np.argwhere(np.logical_not(I)).flatten()
        x = np.zeros(N)
        v = np.zeros(N)
        v_norm = np.append(np.random.normal(0,1,size = len(index)),np.random.normal(0,1,size = N-len(index)))
        v[index] = (v_norm[0:len(index)] + 10)
        v[index_not] = (v_norm[len(index):]-10)
    if level_selection or correlated_time_test:
        x = np.ones(N)
        v_norm = np.random.normal(0,1,size=N)
        v = mu(x) + sigma(x)*v_norm
    elif radiative_transport:
        x = test1(N); v = np.random.uniform(-1,1,size=N)/epsilon
        v_norm = v/sigma(x)
    elif (density_est and post_collisional) or (ml_test_KD and post_collisional) or (ml_test_APS and post_collisional) or (diffusion_limit and post_collisional):
        x,v,v_norm = test3(N)
        v = mu(x)+ sigma(x)*v_norm#
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
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
    elif (density_est and post_collisional) or (ml_test_KD and post_collisional) or (ml_test_APS and post_collisional) or (diffusion_limit and post_collisional):
        x = np.ones(N); v_norm = np.random.normal(0,1,size=N)
        x,v,v_norm = test3(N)
        # v = v_norm.copy()
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        x,v,v_norm = test2(N)
        v=v_norm.copy()
        # x=np.ones(N)*3
    elif correlation_test or correlated_time_test:
        x = np.ones(N); v = np.random.normal(0,1,size=N)
        v_norm = v.copy()
    return x,v,v_norm

#sets the collision rate
def R(x):
    if radiative_transport or correlated_time_test or one_step_dist:
        return 1/(epsilon**2)
    elif level_selection:
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        return 1/epsilon**2*((-a*(x-0.5)+b)*(x<=0.5) + (a*(x-0.5)+b)*(x>0.5))

def r(x):
    if ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        return (-a*(x-0.5)+b)*(x<=0.5) + (a*(x-0.5)+b)*(x>0.5)
    else:
        return np.ones(len(x))

def dR(x):
    if radiative_transport or correlated_time_test or one_step_dist:
        return np.zeros(x.size)
    elif level_selection:
        return (x<=1)*(-b*a) + (x>1)*(b*a)
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        if a==0:
            return np.zeros(x.size)
        else:
            return -a/epsilon**2*(x<=0.5) + a/epsilon**2*(x>0.5)

#Anti derivative of R
def R_anti(x):
    if radiative_transport or correlated_time_test or one_step_dist:
        return x/(epsilon**2)
    elif level_selection:
        return (-b*a/2*x**2+b*(a+1)*x)*(x<=1) + (-b*a/2+b*(a+1) -b*a/2-b*(1-a)+b*a/2*x**2+b*(1-a)*x)*(x>1)
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        return 1/epsilon**2*((-a*(x**2/2-0.5*x)+b*x)*(x<=0.5) + ((-a*(0.5**2/2-0.5*0.5)+b*0.5) + (a*(x**2/2-0.5*x)+b*x) - (a*(0.5**2/2-0.5*0.5)+b*0.5))*(x>0.5))

#Sample Collision
# @njit(nogil=True,parallel = True)
def SC(x,v,e):
    if radiative_transport or a==0 or correlated_time_test or one_step_dist:
        dtau = 1/R(x)*e
    elif level_selection and False:# or density_est or ml_test_APS or ml_test_KD:
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
    elif ml_test_KD or ml_test_APS or density_est or level_selection or diffusion_limit:
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
            n = x.size
            dtau = np.zeros(n);
            #integral value to add to remainder of rhs
            I = np.zeros(n);
            #remainder of rhs. Changes when particle moves across boundary
            # e_remainder = np.abs(e*v)
            e_local = e.copy()
            #active paths. Meaning they still need to determine tau
            A = np.ones(n).astype(np.bool_)
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
                    temp = np.zeros(n).astype(np.bool_); temp[index] =True# np.put(temp,index,1)
                    # print('temp')
                    # print(temp)
                else:
                    temp = np.zeros(n).astype(np.bool_)
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

# @njit(nogil=True,parallel=True)
def roots(x,v,e):
    if density_est or ml_test_APS or ml_test_KD or level_selection or diffusion_limit:
        alpha = -a/epsilon**2*(x<=0.5) + a/epsilon**2*(x>0.5)
        beta = (b+a/2)/epsilon**2*(x<=0.5) + (b-a/2)/epsilon**2*(x>0.5)
        pc = -e*v; pb = alpha*v*x+beta*v; pa = alpha/2*v**2
        p = np.concatenate((pc.reshape((1,e.size)),pb.reshape((1,e.size)),pa.reshape((1,e.size))),axis=0).T
        r = np.zeros_like(x)
        for i in prange(x.size):
            # r[i] = np.polynomial.polynomial.polyroots(p[i,:])
            r[i] = np.roots(np.flip(p[i,:]))[1]
        return r

# @njit(nogil=True,parallel = True)
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
    if ml_test_KD or ml_test_APS or density_est or correlated_time_test or diffusion_limit or one_step_dist:
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
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm#/epsilon
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = mu(x) + sigma(x)*v_norm
    return v_next,v_norm

def M_nu(x):
    '''Distribution of velocity NOT scaled by epsilon, i.e. nu'''
    if radiative_transport:
        v_next = np.random.uniform(-1,1,size=len(x))
        v_norm = v_next.copy()
    elif goldstein_taylor:
        U = np.random.uniform(0,1,size=len(x))
        v_next =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)
        v_norm = v_next.copy()
    elif ml_test_KD or ml_test_APS or density_est or diffusion_limit:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = v_norm.copy()
    else:
        v_norm = np.random.normal(0,1,size=x.size)
        v_next = v_norm.copy()
    return v_next,v_norm

def boundary_periodic(x):
    x0 = 0; xL = 1
    l = xL-x0 #Length of x domain
    I_low = (x<x0); I_high = (x>xL);
    x_new = x.copy()
    x_new[I_low] = xL-((x0-x[I_low])%l)
    x_new[I_high] = x0 + ((x[I_high]-xL)%l)
    return x_new

def boundary(x):
    return x

v_ms = 1

'''Function related to the quantity of interest, E(F(X,V))'''
def F(x,v=0):
    if goldstein_taylor:
        return x**2
    else:
        return x


jit_module(nopython=True,nogil=True)


if __name__ == '__main__':
    if args.folder:
        path = os.getcwd()
        try:
            sep = args.separator if args.separator else '\\'
            path = path+ sep+args.folder
            os.chdir(path)
        except FileNotFoundError:
            print('Maybe the separator is wrong. Give the separator that fits with your operating system. It is done by adding the argument -sep "your_separator"')
            raise
    if ml_cost:

        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_for_a={a}_b={b}_epsilon={epsilon}.txt')
        plt.figure()
        plt.loglog(dt_list[1:],cost2[1:], label='APS')
        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_rev_True_diff_True_std_for_a={a}_b={b}_epsilon={epsilon}.txt')
        plt.loglog(dt_list[1:],cost2[1:], label='RAPSD w. standard correlation')
        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_rev_True_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt')
        plt.loglog(dt_list[1:],cost2[1:], label='RAPSD w. weighted correlation')
        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_rev_False_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt')
        plt.loglog(dt_list[1:],cost2[1:], label='APSD')
        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_KD_for_a={a}_b={b}_epsilon={epsilon}.txt')
        plt.loglog(dt_list[1:],cost2[1:], label='KD')
        plt.legend()
        plt.xlim(max(dt_list),min(dt_list))
        plt.xlabel(r'$\Delta t$')
        plt.ylabel('Cost per path')
        plt.show()

        E2=1/2**np.arange(10,22)
        start = 0
        cost = []
        for e2 in E2[start:]:
            df = pd.read_csv(f'resultfile_complexity_{e2}_APS_for_a={a}_b={b}_epsilon={epsilon}.txt')
            print(df)
            L = len(df.get('N C'))-1
            cost += [df.get('N C')[L]]
        plt.figure()
        plt.plot(np.sqrt(E2[start:]),cost,label=r'APS')

        cost = []
        for e2 in E2[start:]:
            df = pd.read_csv(f'resultfile_complexity_{e2}_APS_rev_True_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt')
            L = len(df.get('N C'))-1
            cost += [df.get('N C')[L]]
        plt.plot(np.sqrt(E2[start:]),cost,label=r'RAPSD w. weighted correlation')
        c4 = np.mean(cost*E2[start:]/np.log(np.sqrt(E2[start:]))**2)
        c4 *= 6
        plt.plot(np.sqrt(E2[start:]),c4*np.log(np.sqrt(E2[start:]))**2/E2[start:],label= r'$\mathcal{O}(\log(E)^2E^{-2})$')
        cost = []
        for e2 in E2[start:]:
            df = pd.read_csv(f'resultfile_complexity_{e2}_APS_rev_True_diff_True_std_for_a={a}_b={b}_epsilon={epsilon}.txt')
            L = len(df.get('N C'))-1
            cost += [df.get('N C')[L]]
        plt.plot(np.sqrt(E2[start:]),cost,label=r'RAPSD w. standard correlation')

        cost = []
        for e2 in E2[start:]:
            df = pd.read_csv(f'resultfile_complexity_{e2}_APS_rev_False_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt')
            L = len(df.get('N C'))-1
            cost += [df.get('N C')[L]]
        plt.plot(np.sqrt(E2[start:]),cost,label=r'APSD')

        cost = []
        for e2 in E2[start:]:
            df = pd.read_csv(f'resultfile_complexity_{e2}_KD_for_a={a}_b={b}_epsilon={epsilon}.txt')
            # print(dfs[e2])
            L = len(df.get('N C'))-1
            cost += [df.get('N C')[L]]
        # CE2 = cost*E2
        c4 = np.mean(cost*E2[start:])
        c4 *= 1.5
        plt.plot(np.sqrt(E2[start:]),c4/E2[start:],label= r'$\mathcal{O}(E^{-2})$')
        plt.plot(np.sqrt(E2[start:]),cost,label=r'KD')

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$E$')
        plt.ylabel(r'$Cost$')
        plt.legend()
        plt.show()
    if ml_test_APS:
        E2=1/2**np.arange(10,22)
        if uf:
            if type == 'conv' or type == 'both':
                if not rev and not diff:
                    (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_for_a={a}_b={b}_epsilon={epsilon}.txt')
                else:
                    if std:
                        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_rev_{rev}_diff_{diff}_std_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    else:
                        (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_APS_rev_{rev}_diff_{diff}_for_a={a}_b={b}_epsilon={epsilon}.txt')

                '''Find index where they transition'''
                i = 0
                old = 0
                for va in var2[1:]:
                    if va < old:
                        break
                    i+=1
                    old=va
                alpha = -np.polyfit(range(i+3,var2.size),np.log2(np.abs(bias[i+3:])),1)[0]
                beta = -np.polyfit(range(i+3,var2.size),np.log2(np.abs(var2[i+3:])),1)[0]
                gamma = np.polyfit(range(i+3,var2.size),np.log2(np.abs(cost2[i+3:])),1)[0]
                print(f'alpha= {alpha}, beta= {beta}, gamma= {gamma}')
                '''Complexity test based on numercial estimations'''
                dfs = {}
                pd.set_option('display.float_format', '{:.2E}'.format)
                for e2 in E2:
                    L=4 + i
                    E = np.append(v[i],bias[i:L])
                    V = np.append(var1[i],var2[i:L])
                    C = np.append(cost1[i],cost2[i:L])
                    N = np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C)))
                    test = (np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2))
                    out=False
                    while not test:
                        L+=1
                        # print(L)
                        if L>=v.size:
                            out = True
                            print(f'Warning!. Need more results to adhere to bias requirement for e2= {e2}')
                            break
                        N = np.append(N,0)
                        E = np.append(E,bias[L])
                        V = np.append(V,var2[L])
                        C = np.append(C,cost2[L])
                        N += np.maximum(0,np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64))
                        test = (np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2))
                    if not out:
                        df = pd.DataFrame({'E':E,'V':V,'V[Y]':V/N,'C':C,'N':N,'N C':C*N,'dt':dt_list[i-1:min(L,v.size-1)]})
                        df = df.append(pd.DataFrame({'E':np.sum(E),'V':' ','V[Y]':[np.sum(V/N)],'C':[np.sum(C)],'N':' ','N C':[np.dot(C,N)],'dt':' '}),ignore_index=True)
                        # if not rev and not diff:
                        #     df.to_csv(f'resultfile_complexity_{e2}_APS_for_a={a}_b={b}_epsilon={epsilon}.txt',index=False)
                        # else:
                        #     if std:
                        #         df.to_csv(f'resultfile_complexity_{e2}_APS_rev_{rev}_diff_{diff}_std_for_a={a}_b={b}_epsilon={epsilon}.txt',index=False)
                        #     else:
                        #         df.to_csv(f'resultfile_complexity_{e2}_APS_rev_{rev}_diff_{diff}_for_a={a}_b={b}_epsilon={epsilon}.txt',index=False)
                        print(df)
                        dfs[e2] = df
                    else:
                        break

                fig,ax = plt.subplots()
                ax.plot(dt_list[1:],var2[1:],':',label='var(F(X^f)-F(X^c))')
                ax.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
                ax.plot(dt_list[1:],np.abs(bias[1:]),':',label='|mean(F(X^f)-F(X^c))|')
                ax.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
                ax.set_title(f'Plot of variance and bias')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(max(dt_list),min(dt_list))
                ax.set_xlabel(r'$\Delta t$')
                plt.legend()
                fig2,ax2 = plt.subplots()
                ax2.plot(dt_list[1:],cost2[1:],':')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_xlim(max(dt_list),min(dt_list))
                ax2.set_xlabel(r'$\Delta t$')
                ax2.set_ylabel(r'Wall clock time per path')
                # plt.legend()
                # plt.savefig(f'var_APS_diff_{diff}_rev_{rev}_std_{std}_eps_{epsilon}_a_{a}_b_{b}.png')
                plt.figure()
                plt.plot(range(1,dt_list.size),kur1[1:],':')
                plt.xlabel('Levels')
                plt.ylabel(r'$\frac{E[(F(Y^{\Delta t_l})-F(Y^{\Delta t_{l-1}}))^4]}{E[(F(Y^{\Delta t_l})-F(Y^{\Delta t_{l-1}}))^2]^2}$')
                plt.title(f'Plot of kurtosis')
                plt.figure()
                plt.plot(range(1,dt_list.size),cons[1:],':')
                plt.title(f'Plot check of consistency')
                plt.ylabel(r'$\frac{|a-b+c|}{3(\sqrt{V[a]}+\sqrt{V[b]}+\sqrt{V[c]})}$')
                plt.xlabel('Levels')
                plt.show()
            if type == 'comp' or type == 'both':
                dfs = {}
                cost = []
                start = 5
                for e2 in E2[start:]:
                    if not rev and not diff:
                        dfs[e2] = pd.read_csv(f'resultfile_complexity_{e2}_APS_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    else:
                        if std:
                            dfs[e2] = pd.read_csv(f'resultfile_complexity_{e2}_APS_rev_{rev}_diff_{diff}_std_for_a={a}_b={b}_epsilon={epsilon}.txt')
                        else:
                            dfs[e2] = pd.read_csv(f'resultfile_complexity_{e2}_APS_rev_{rev}_diff_{diff}_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    print(dfs[e2])
                    L = len(dfs[e2].get('N_l C_l'))-1
                    cost += [dfs[e2].get('N_l C_l')[L]]
                c4 = np.mean(cost*E2[start:]/np.log(np.sqrt(E2[start:])))
                c4 += 4*c4
                plt.plot(E2[start:],c4*np.log(np.sqrt(E2[start:]))/E2[start:],label= r'$\mathcal{O}(\frac{\log(\sqrt{E})^2}{E^{2}})$')
                plt.plot(E2[start:],cost,label=r'Total cost')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                plt.show()
            if type!= 'comp' and type != 'conv' and type != 'both':
                sys.exit('ERROR: Invalid input. When asked, you should give either of these three inputs: "comp", "conv" or "both" without apostrophe.')

        else:
            if N is None:
                N = 120_000
            N0=2000; T=1; dt_list = T/2**np.arange(0,17,1); t0=0; M_t=2
            if args.save_file:
                '''file names:
                "logfile_APS_for_a={a}_b={b}_epsilon={epsilon}.txt"
                "logfile_APS_rev_True_diff_False_for_a={a}_b={b}_epsilon={epsilon}.txt"
                "logfile_APS_rev_False_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt"
                "logfile_APS_rev_True_diff_True_for_a={a}_b={b}_epsilon={epsilon}.txt"
                '''
                if not rev and not diff and not std:
                    logfile = open(f'logfile_APS_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
                elif (rev or diff) and not std:
                    logfile = open(f'logfile_APS_rev_{rev}_diff_{diff}_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
                else:
                    logfile = open(f'logfile_APS_rev_{rev}_diff_{diff}_std_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
            else:
                logfile=None
            # start = time.perf_counter()
            # x = SMC(dt,t0,T,N,eps,Q,M,r,rev=rev,diff=diff,v_ms=1)
            if type=='comp':
                APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile,complexity=True,convergence=False,rev=rev,diff=diff,v_ms=v_ms,std=std,)
            elif type=='conv':
                APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile,complexity=False,convergence=True,rev=rev,diff=diff,v_ms=v_ms,std=std)
            elif type=='both':
                APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile,complexity=True,convergence=True,rev=rev,diff=diff,v_ms=v_ms,std=std)
            else:
                sys.exit('ERROR: Invalid input. When asked, you should give either of these three inputs: "comp", "conv" or "both" without apostrophe.')
    if ml_test_KD:
        E2=1/2**np.arange(10,22)
        if uf:
            if type == 'conv' or type == 'both':
                (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_KD_for_a={a}_b={b}_epsilon={epsilon}.txt')
                '''Find index where they transition'''
                i = 0
                old = 0
                for va in var2[1:]:
                    if va < old:
                        break
                    i+=1
                    old=va
                alpha = -np.polyfit(range(i+3,var2.size),np.log2(np.abs(bias[i+3:])),1)[0]
                beta = -np.polyfit(range(i+3,var2.size),np.log2(np.abs(var2[i+3:])),1)[0]
                gamma = np.polyfit(range(i+3,var2.size),np.log2(np.abs(cost2[i+3:])),1)[0]
                print(f'alpha= {alpha}, beta= {beta}, gamma= {gamma}')
                '''Complexity test based on numercial estimations'''
                dfs = {}
                pd.set_option('display.float_format', '{:.2E}'.format)
                for e2 in E2:
                    L=4 + i
                    E = np.append(v[i],bias[i:L])
                    V = np.append(var1[i],var2[i:L])
                    C = np.append(cost1[i],cost2[i:L])
                    N = np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C)))
                    test = (np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2))
                    out=False
                    while not test:
                        L+=1
                        if L>=v.size:
                            out = True
                            print(f'Warning!. Need more results to adhere to bias requirement for e2= {e2}')
                            break
                        N = np.append(N,0)
                        E = np.append(E,bias[L])
                        V = np.append(V,var2[L])
                        C = np.append(C,cost2[L])
                        N += np.maximum(0,np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64))
                        test = (np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2))
                    if not out:
                        df = pd.DataFrame({'E':E,'V':V,'V[Y]':V/N,'C':C,'N':N,'N C':C*N,'dt':dt_list[i-1:min(L,v.size-1)]})
                        df = df.append(pd.DataFrame({'E':np.sum(E),'V':' ','V[Y]':[np.sum(V/N)],'C':[np.sum(C)],'N':' ','N C':[np.dot(C,N)],'dt':' '}),ignore_index=True)
                        # df.to_csv(f'resultfile_complexity_{e2}_KD_for_a={a}_b={b}_epsilon={epsilon}.txt',index=False)
                        dfs[e2] = df
                        print(df)
                    else:
                        break
                print(L)
                '''Plotting number of paths and computational costs as functions of MSE'''
                fig, (ax1, ax2) = plt.subplots(2, 1)
                N = []
                Cost = []
                end = len(dfs)
                start = end-6
                for e2 in E2[start:end]:
                    N += [dfs[e2]['N'][:-1]]
                    Cost += [dfs[e2]['N C'][N[-1].size]]
                    ax1.plot(range(N[-1].size),N[-1],label=r'$E^2 = ${:.2E}'.format(e2))
                ax2.loglog(E2[start:end],Cost)
                c4 = np.mean(Cost*E2[start:end])
                c4 *= 2
                ax2.plot(E2[start:end],c4*1/E2[start:end],label= r'$\mathcal{O}(E^{-2})$')
                ax1.set_yscale('log')
                ax2.set_xlabel(r'$E^2$')
                ax2.set_ylabel(r'Total costs')
                ax1.set_xlabel(r'Levels')
                ax1.set_ylabel(r'Paths')
                ax1.legend()
                plt.show()


                fig,ax = plt.subplots()
                ax.plot(dt_list[1:],var2[1:],':',label='var(F(X^f)-F(X^c))')
                ax.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
                ax.plot(dt_list[1:],np.abs(bias[1:]),':',label='|mean(F(X^f)-F(X^c))|')
                ax.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
                ax.set_title(f'Plot of variance and bias')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(max(dt_list),min(dt_list))
                ax.set_xlabel(r'$\Delta t$')
                plt.legend()
                fig2,ax2 = plt.subplots()
                ax2.plot(dt_list[1:],cost2[1:],':')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_xlim(max(dt_list),min(dt_list))
                ax2.set_xlabel(r'$\Delta t$')
                ax2.set_ylabel(r'Wall clock time per path')
                # plt.legend()
                # plt.savefig(f'var_KD_eps_{epsilon}_a_{a}_b_{b}.png')
                plt.figure()
                plt.plot(range(1,dt_list.size),kur1[1:],':')
                plt.xlabel('Levels')
                plt.ylabel(r'$\frac{E[(F(Y^{\Delta t_l})-F(Y^{\Delta t_{l-1}}))^4]}{E[(F(Y^{\Delta t_l})-F(Y^{\Delta t_{l-1}}))^2]^2}$')
                plt.title(f'Plot of kurtosis')
                plt.figure()
                plt.plot(range(1,dt_list.size),cons[1:],':')
                plt.ylabel(r'$\frac{|a-b+c|}{3(\sqrt{V[a]}+\sqrt{V[b]}+\sqrt{V[c]})}$')
                plt.xlabel('Levels')
                plt.title(f'Plot check of consistency')
                plt.show()
            if type == 'comp' or type == 'both':
                dfs = {}
                cost = []
                start=1
                for e2 in E2[start:]:
                    dfs[e2] = pd.read_csv(f'resultfile_complexity_{e2}_KD_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    # print(dfs[e2])
                    L = len(dfs[e2].get('N_l C_l'))-1
                    cost += [dfs[e2].get('N_l C_l')[L]]
                # CE2 = cost*E2
                c4 = np.mean(cost*E2[start:])
                c4 *= 3
                plt.plot(E2[start:],c4/E2[start:],label= r'$\mathcal{O}(E^{-2})$')
                plt.plot(E2[start:],cost,label=r'Total cost')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                plt.show()
            if type!= 'comp' and type != 'conv' and type != 'both':
                sys.exit('ERROR: Invalid input. When asked, you should give either of these three inputs: "comp", "conv" or "both" without apostrophe.')

        else:
            if N is None:
                N = 120_000
            N0=32; T=1; dt_list = T/2**np.arange(0,17,1); t0=0
            if args.save_file:
                logfile = open(f'logfile_KD_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
            else:
                logfile=None
            if type=='comp':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=True,convergence=False)
            elif type=='conv':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=False,convergence=True)
            elif type=='both':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=True,convergence=True)
            else:
                sys.exit('ERROR: Invalid input. When asked, you should give either of these three inputs: "comp", "conv" or "both" without apostrophe.')
    if goldstein_taylor:
        if N is None:
            N=120_000;
        N0=40; M_t = 2; t0=0;T=0.5
        dt_list = T/2**np.arange(0,17,1); E2=np.array([0.01,0.0001,1e-6],dtype=np.float64)
        if args.save_file:
            logfile = open(f'logfile_APS_Goldstein_Taylor_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
        else:
            logfile=None
        APML_test(N,N0,dt_list,E2,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,logfile,complexity=True,convergence=False)



        # x_std = SMC_par((T-t0)/2**19,t0,T,N,epsilon,Q_nu,M_nu,boundary,r)
        # W,err=APSMC_density_test(dt_list,M_t,t0,T,N/10,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std)
        # plt.errorbar(dt_list,W,err,label='Error APS GT dist')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel(r'$\Delta t$')
        # plt.ylabel('Wasserstein distance')
        # plt.legend()
        # plt.show()
    if density_est:
        if N is None:
            N = 1_200_000
        if N%8!=0 or N%80!=0:
            sys.exit('Please provide a number of paths divisible by 8 and 80')
        N0=40; T=1; dt_list = T/2**np.arange(0,17,1); t0=0; M_t=2
        if False:
            print('Testing consistency')
            x_KD=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary)
            dist = pd.DataFrame(data={'x':x_KD,'Method':['KD' for _ in range(N)]})
            print('Done with KMC')
            x_AP = SMC_par((T-t0)/2**12,t0,T,N,epsilon,Q_nu,M_nu,boundary,r)
            print(f'x_ap^2: {np.mean(x_AP**2)}, x_KD^2 = {np.mean(x_KD**2)}')
            dist = dist.append(pd.DataFrame(data={'x':x_AP,'Method':['Splitting' for _ in range(N)]}))
            print(wasserstein_distance(x_AP,x_KD))
            sns.kdeplot(data=dist, x="x",hue='Method',cut=0,common_norm=False)
            plt.show()
            if args.save_file:
                if post_collisional:
                    np.savetxt(f'density_exact_AP_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt',x_AP)
                    np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt',x_KD)
                else:
                    np.savetxt(f'density_exact_AP_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt',x_AP)
                    np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt',x_KD)

        else:
            if uf:
                if post_collisional:
                    x_std = np.loadtxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt')
                    data = np.loadtxt(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt')
                    cost = np.loadtxt(f'density_resultfile_cost_for_a={a}_b={b}_epsilon={epsilon}_post.txt')
                else:
                    # x_std = np.loadtxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    data = np.loadtxt(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt')
                    cost = np.loadtxt(f'density_resultfile_cost_for_a={a}_b={b}_epsilon={epsilon}.txt')
            else:
                print(f'{N} paths used for simulation')
                x0,v0,_ = Q_nu(N)
                x_std=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0,v0/epsilon)
                cost = np.zeros((5,dt_list.size))
                # x_std = np.loadtxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt')
                if args.save_file:
                    if post_collisional:
                        np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt',x_std)
                    else:
                        np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt',x_std)
            if uf:
                W = data[0]
                err = data[1]
            else:
                print('Done with exact')
                # print(f'{N} paths used to estimate density with APSMC and KDMC')
                W,err,cost[0,:]=APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,v_ms=v_ms,x0=x0,v0=v0)
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt','w') as file:
                        np.savetxt(file,(W,err))
                else:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','w') as file:
                        np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='APS')
            if uf:
                W = data[2]
                err = data[3]
            else:
                start = time.time()
                W,err,cost[1,:]=APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,rev=True,x0=x0,v0=v0)
                print(f'APS with reverse one-step method is done. Time: {time.time()-start}')
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt','a') as file:
                        np.savetxt(file,(W,err))
                else:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','a') as file:
                        np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='RAPS')
            if uf:
                W = data[4]
                err = data[5]
            else:
                W,err,cost[2,:] = KDMC_density_test(dt_list,Q,t0,T,N,mu,sigma,M,R,SC,dR=dR,boundary=boundary,x_std=x_std,x0=x0,v0=v0/epsilon)
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt','a') as file:
                        np.savetxt(file,(W,err))
                else:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','a') as file:
                        np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='KD')
            if uf:
                W = data[6]
                err = data[7]
            else:
                start = time.time()
                W,err,cost[3,:]=APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,rev=False,diff=True,v_ms=v_ms,x0=x0,v0=v0)
                print(f'APS with altered diffusive coefficient is done. Time: {time.time()-start}')
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt','a') as file:
                        np.savetxt(file,(W,err))
                else:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','a') as file:
                        np.savetxt(file,(W,err))
            plt.errorbar(dt_list,W,err,label='APSD')
            if uf:
                W = data[8]
                err = data[9]
            else:
                start = time.time()
                W,err,cost[4,:]=APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,rev=True,diff=True,x0=x0,v0=v0)
                print(f'APS with reverse one-step method and altered diffusive coeficient is done. Time: {time.time()-start}')
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt','a') as file:
                        np.savetxt(file,(W,err))
                    with open(f'density_resultfile_cost_for_a={a}_b={b}_epsilon={epsilon}_post.txt','w') as file:
                        np.savetxt(file,cost)
                else:
                    with open(f'density_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt','a') as file:
                        np.savetxt(file,(W,err))
                    with open(f'density_resultfile_cost_for_a={a}_b={b}_epsilon={epsilon}.txt','w') as file:
                        np.savetxt(file,cost)
            plt.errorbar(dt_list,W,err,label='RAPSD')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Delta t$')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.figure()
            names = ['APS','RAPS','KD','APSD','RAPSD']
            for c,n in zip(cost,names):
                plt.plot(dt_list[1:],c[1:],label=n)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\Delta t$')
            plt.ylabel('Wall clock time')
            plt.legend()
            plt.show()
    if radiative_transport:
        '''Numerical experiemnt on radiative transport with
        rho(x,0) = 1+cos(2*pi*(x+0.5)) and V ~ U(-1,1)'''
        x_lim = (0,1); v_lim = (-1,1)
        N = 400_000
        dt =0.1/2**19 ;t0=0;T=0.1
        x = APSMC_par(dt,t0,T,N,epsilon,Q_nu,M_nu,boundary_periodic,r)
        print(f'Done with APSMC, x^2 = {np.mean(x**2)}')
        dist = pd.DataFrame(data={'x':x,'Method':['APS' for _ in range(N)]})
        x = KDMC_par(dt,N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary_periodic)
        print(f'Done with KDMC, x^2 = {np.mean(x**2)}')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['KD' for _ in range(N)]}))
        x = SMC_par(dt,t0,T,N,epsilon,Q_nu,M_nu,boundary_periodic,r)
        print(f'Done with SMC, x^2={np.mean(x**2)}')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['SS' for _ in range(N)]}))
        x=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary_periodic)
        print(f'Done with KMC, x^2={np.mean(x**2)}')
        dist = dist.append(pd.DataFrame(data={'x':x,'Method':['Kinetic' for _ in range(N)]}))
        sns.kdeplot(data=dist, x="x",hue='Method',cut=0,common_norm=False)
        plt.show()
    if correlation_test:
        dt_f = 0.2; M_t = 5; T=10; t0=0
        correlated_test(dt_f,M_t,t0,T,epsilon,1,Q_nu,M_nu,r=r,plot=True,plot_var=False,rev = True)
    if correlated_time_test:
        if not args.paths:
            N=120_000
        print(f'Test is done with {N} paths')
        T = 1; t0=0; M_t = 2
        dt_list = T/2**np.arange(0,17,1)
        if uf:
            '''a=1,b=10'''
            times_numba=np.loadtxt(f'correlation_time_results_eps_{epsilon}_numba.txt')
            times=np.loadtxt(f'correlation_time_results_eps_{epsilon}.txt')
        else:
            times = np.empty((2,8))
            print('compile functions')
            KD_conv(8,T/2**np.arange(0,2,1),Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary)
            APS_conv(8,T/2**np.arange(0,2,1),Q_nu,t0,T,M_t,epsilon,M_nu,r,F,boundary,strategy=1)
            print('Compilation done!')
            # for i in range(1,dt_list.size):
            #     start = time.time()
            #     x0,v0,v_l1_next = Q(N)
            #     KD_conv(N,dt_list[i-1:i],Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary)
            #     times[0,i-1] = time.time()-start
            #     start = time.time()
            #     APS_conv(N,dt_list[i-1:i],Q_nu,t0,T,M_t,epsilon,M_nu,r,F,boundary,strategy=1)
            #     times[1,i-1] = time.time()-start
            for i,n in enumerate(12_000*np.arange(1,9)):
                N=int(n)
                print(N)
                start = time.perf_counter()
                _,_,_,_,_,_,_,_,_,_,cost1,cost2 = KD_conv(N,dt_list,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary)
                times[0,i] = time.perf_counter() -start
                # times[1,i] = np.sum(cost2)
                start = time.perf_counter()
                _,_,_,_,_,_,_,_,_,_,cost1,cost2 = APS_conv(N,dt_list,Q_nu,t0,T,M_t,epsilon,M_nu,r,F,boundary,strategy=1)
                times[1,i] = time.perf_counter()-start
                print(times[:,i])
                # times[3,i] = np.sum(cost2)
        if args.save_file:
            np.savetxt(f'correlation_time_results_eps_{epsilon}.txt',times)
        plt.plot(12_000*np.arange(1,9),times[0,:]/times_numba[0,:],label='KD ')
        plt.plot(12_000*np.arange(1,9),times[1,:]/times_numba[1,:],label='APS')
        # plt.plot(8_000*np.arange(1,11),times[2,:]/times_numba[2,:],label='APSMC')
        # plt.plot(8_000*np.arange(1,11),times[3,:]/times_numba[3,:],label='APSML')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Number of paths')
        plt.ylabel('Speed-up ratio')
        plt.legend()
        plt.show()
    if level_selection:
        E2=0.01/2**np.arange(0,13)
        if uf:
            (dt_list,v,bias,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(f'resultfile_KD_level_selection_for_a={a}_b={b}_epsilon={epsilon}.txt')
            plt.plot(dt_list[1:],var2[1:],':',label='var(F(x^f)-F(X^c))')
            plt.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
            plt.plot(dt_list[1:],np.abs(bias[1:]),':',label='mean(|F(x^f)-F(X^c)|)')
            plt.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
            plt.title(f'Plot of variance and bias')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.figure()
            plt.plot(range(1,dt_list.size),kur1[1:],':')
            plt.title(f'Plot of kurtosis')
            plt.figure()
            plt.plot(range(1,dt_list.size),cons[1:],':')
            plt.title(f'Plot check of consistency')
            # dfs = {}
            # for e2 in E2:
            #     dfs[e2] = pd.read_csv(f'resultfile_complexity_{e2}_KD_for_a={a}_b={b}_epsilon={epsilon}.txt')
            #     print(dfs[e2])
            plt.show()

        else:
            if N is None:
                N = 120_000
            N0=16; T=1; dt_list = T/2**np.arange(0,17,1) if a==0 else T/2**np.arange(0,17,1); E2=0.01/2**np.arange(0,13); t0=0
            if args.save_file:
                logfile = open(f'logfile_KD_level_selection_for_a={a}_b={b}_epsilon={epsilon}.txt','w')
            else:
                logfile=None
            if type=='comp':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=True,convergence=False)
            elif type=='conv':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=False,convergence=True)
            elif type=='both':
                KDML_test(N,N0,dt_list,E2,epsilon,Q,t0,T,mu,sigma,M,R,SC,F,logfile,R_anti=R_anti,dR=dR,complexity=True,convergence=True)
            else:
                sys.exit('ERROR: Invalid input. When asked, you should give either of these three inputs: "comp", "conv" or "both" without apostrophe.')
    if diffusion_limit:
        '''
        epsilon= [1,0.32,0.1,0.032,0.01,0.005]
        a = 0, b = 1
        '''
        T = 1;t0=0;dt_list=T/2**np.arange(0,7);M_t=2;
        if uf:
            eps = np.array([1,0.32,0.1,0.032])#,0.01,0.005])
            step=int(input(f'Give index of step size to plot for: {dt_list}\n')) #between 0 and 6
            if args.folder == 'nu_standard' and not post_collisional:
                data = np.loadtxt(f'density_resultfile_a_{a}_b_{b}_all_eps_and_dt.txt')
                # length is number of epsilons
                W1 = np.zeros(6); err1 = np.zeros(6)
                W2 = np.zeros(6); err2 = np.zeros(6)
                W3 = np.zeros(6); err3 = np.zeros(6)
                W4 = np.zeros(6); err4 = np.zeros(6)
                W5 = np.zeros(6); err5 = np.zeros(6)
                for i in range(6):
                    W1[i] = data[i*10,step]; err1[i] = data[5+i*10,step]
                    W2[i] = data[1+i*10,step]; err2[i] = data[6+i*10,step]
                    W3[i] = data[2+i*10,step]; err3[i] = data[7+i*10,step]
                    W4[i] = data[3+i*10,step]; err4[i] = data[8+i*10,step]
                    W5[i] = data[4+i*10,step]; err5[i] = data[9+i*10,step]
            else:

                W1 = np.zeros(eps.size); err1 = np.zeros(eps.size); c1 = np.zeros(eps.size)
                W2 = np.zeros(eps.size); err2 = np.zeros(eps.size); c2 = np.zeros(eps.size)
                W3 = np.zeros(eps.size); err3 = np.zeros(eps.size); c3 = np.zeros(eps.size)
                W4 = np.zeros(eps.size); err4 = np.zeros(eps.size); c4 = np.zeros(eps.size)
                W5 = np.zeros(eps.size); err5 = np.zeros(eps.size); c5 = np.zeros(eps.size)
                for i,e in enumerate(eps):
                    if post_collisional:
                        data =  np.loadtxt(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{e}_post.txt')
                    else:
                        data =  np.loadtxt(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{e}.txt')
                        cost =  np.loadtxt(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{e}_cost.txt')
                    W1[i] = data[0,step]; err1[i] = data[5,step]
                    W2[i] = data[1,step]; err2[i] = data[6,step]
                    W3[i] = data[2,step]; err3[i] = data[7,step]
                    W4[i] = data[3,step]; err4[i] = data[8,step]
                    W5[i] = data[4,step]; err5[i] = data[9,step]
                    c1[i] = cost[0,step]
                    c2[i] = cost[1,step]
                    c3[i] = cost[2,step]
                    c4[i] = cost[3,step]
                    c5[i] = cost[4,step]
            print(f'Errors for APS: {W1}')
            print(f'Standard deviation of error: {err1}')

            plt.errorbar(eps,W1,err1,label='APS')
            plt.errorbar(eps,W2,err2,label='KD')
            plt.errorbar(eps,W3,err3,label='APSD')
            plt.errorbar(eps,W4,err4,label='RAPSD')
            plt.errorbar(eps,W5,err5,label='RAPS')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\epsilon$')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.figure()
            names = ['APS','RAPS','KD','APSD','RAPSD']
            for c,n in zip([c1,c2,c3,c4,c5],names):
                plt.plot(eps,c,label=n)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\epsilon$')
            plt.ylabel('Wall clock time')
            plt.legend()
            plt.show()


        else:
            if not args.save_file:
                sys.exit('If new results should be simulated then please save those results with "-sf" command')
            if N is None:
                N = 1_200_000
            print(f'{N} paths used')
            x0,v0,_ = Q_nu(N)
            x_std=KMC_par(N,Q,t0,T,mu,sigma,M,R,SC,dR,boundary,x0,v0/epsilon)
            print('Exact is done')
            # np.savetxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}_post.txt',x_std)
            # x_std = np.loadtxt(f'density_exact_KD_resultfile_for_a={a}_b={b}_epsilon={epsilon}.txt')
            W = np.zeros((5,dt_list.size))
            err = np.zeros((5,dt_list.size))
            cost = np.zeros((5,dt_list.size))
            # test = x_std.copy()
            W[0,:],err[0,:],cost[0,:] = APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,v_ms=v_ms,x0=x0,v0=v0)
            print('APSMC is done')
            # W[0,:],err[0,:],cost[0,:] = KDMC_density_test(dt_list,Q,t0,T,N,mu,sigma,M,R,SC,dR=dR,boundary=boundary,x_std=x_std,x0=x0,v0=v0/epsilon)
            W[1,:],err[1,:],cost[1,:] = KDMC_density_test(dt_list,Q,t0,T,N,mu,sigma,M,R,SC,dR=dR,boundary=boundary,x_std=x_std,x0=x0,v0=v0/epsilon)
            print('KDMC is done')
            W[2,:],err[2,:],cost[2,:] = APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,v_ms=v_ms,diff=True,x0=x0,v0=v0)
            print('APSMC is done, diff=True')
            W[3,:],err[3,:],cost[3,:] = APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,v_ms=v_ms,diff=True,rev=True,x0=x0,v0=v0)
            print('APSMC is done, diff=True, rev=True')
            W[4,:],err[4,:],cost[4,:] = APSMC_density_test(dt_list,M_t,t0,T,N,epsilon,Q_nu,M_nu,r,F,boundary = boundary,x_std=x_std,v_ms=v_ms,diff=False,rev=True,x0=x0,v0=v0)
            print('APSMC is done, diff=False,rev=true')
            if args.save_file:
                if post_collisional:
                    with open(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{epsilon}_post.txt','w') as f:
                        np.savetxt(f,np.vstack((W,err)))
                else:
                    with open(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{epsilon}.txt','w') as f:
                        np.savetxt(f,np.vstack((W,err)))
                    with open(f'density_resultfile_diffusion_limit_a_{a}_b_{b}_eps_{epsilon}_cost.txt','w') as f:
                        np.savetxt(f,cost)

    if one_step_dist:
        if N is None:
            N = 100_000
        dt = 1
        t = 0
        T = 1
        # print(f'tau > T: {np.where(tau>T)}')
        x = KMC_par(N,Q,t,T,mu,sigma,M,R,SC,dR,boundary)
        dist = pd.DataFrame(data={'x':x})
        sns.kdeplot(data=dist, x="x")
        plt.show()
