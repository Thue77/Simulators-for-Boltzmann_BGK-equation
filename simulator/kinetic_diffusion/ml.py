import numpy as np
from .correlated import correlated
from .mc import KDMC
import time
from .AddPaths import delta,x_hat,Sfunc
from numba import njit,jit_module
from numba import prange,objmode
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys

#Uses existing data - for plotting pruposes
def select_levels_data(V,V_d,t0,T,L_max=4):
    '''
    V: array of variances. Length L+1
    V_d: array of variances of bias. Length L
    output:
    let l be in levels[1:]. Then dt^f = 1/2**l.
    for l=levels[0], dt = 1/2**l
    '''
    l = 1
    for i in range(len(V_d)):
        if V_d[i]>V[i+1]:
            l+=l
        else:
            break
    levels = np.array([l-1,l])
    V_min = V_d[l-1]
    for j in range(l,len(V_d)):
        if levels.size>=L_max:
            break
        if V_d[j]<V_min/2:
            levels =np.append(levels,[j+1])
            V_min = V_d[j]
    return levels


@njit(nogil=True,parallel=True)
def warm_up(L,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=None,dR=None,N=100,tau=None):
    dt_list = 1/2**np.arange(0,L+1)
    Q_l = np.zeros(L+1) #Estimates for all possible first levels
    Q_l_L = np.zeros(L) #Estimates for all adjecant biases
    V_l = np.zeros(L+1)
    V_l_L = np.zeros(L)
    C_l = np.zeros(L+1) ##Cost per path for each level
    C_l_L = np.zeros(L)
    x0,v0,v_l1_next = Q(N)
    # e = np.random.exponential(scale=1,size=N)
    # tau = SC(x0,v0,e)
    for l in prange(L+1):
        if l < L:
            with objmode(start1='f8'):
                start1 = time.perf_counter()
            x_f,x_c = correlated(dt_list[l+1],dt_list[l],x0,v0,v_l1_next,t0,T,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR)
            with objmode(end1='f8'):
                end1 = time.perf_counter()
            C_l_L[l] = (end1-start1)/N
            x_dif = F(x_f)-F(x_c)
            Q_l_L[l] = np.mean(x_dif)
            V_l_L[l] = np.var(x_dif)
        with objmode(start2='f8'):
            start2 = time.perf_counter()
        x = KDMC(dt_list[l],N,Q,t0,T,mu,sigma,M,R,SC)
        with objmode(end2='f8'):
            end2 = time.perf_counter()
        C_l[l] = (end2-start2)/N
        est = F(x)
        Q_l[l] = np.mean(est)
        V_l[l] = np.var(est)
    return Q_l,Q_l_L,V_l,V_l_L,C_l,C_l_L


# @njit(nogil=True)
def select_levels(L,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=None,dR=None,N=100,tau=None):
    '''
    V: array of variances. Length L+1
    V_d: array of variances of bias. Length L
    output:
    let l be in levels[1:]. Then dt^f = 1/2**l.
    for l=levels[0], dt = 1/2**l
    '''
    Q_est,Q_d,V,V_d,C,C_d = warm_up(L,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,N)
    l = 1
    for i in range(len(V_d)):
        if V_d[i]>V[i+1]:
            l+=l
        else:
            break
    # test = V_d > V[1:]
    # if np.sum(test)>1:
        # l = np.argwhere(test).flatten()[-1]+2 #Last index where variance of bias is larger than V
    levels = [l-1,l]
    V_min = V_d[l-1]
    for j in range(l,len(V_d)):
        if V_d[j]<V_min/2:
            levels += [j+1]
            V_min = V_d[j]
    L_set = np.array(levels)
    # print(f'Last level: {len(V)-1}, levels: {L_set}')
    # sys.exit()
    '''Set up output variables based on level selection and set values of non adjecant levels to zero'''
    Q_out = np.empty(len(L_set)) #List of ML estimates for each level
    V_out = np.empty(len(L_set)) #Variances of estimates on each level
    C_out = np.empty(len(L_set)) #Cost of estimates on each level
    Q_out[0] = Q_est[levels[0]]; V_out[0] = V[levels[0]]; C_out[0] = C[levels[0]] #Values for first level
    '''Note that len(Q_l_l1)=L and len(Q_l)=L+1. So the first value in Q_l_l1 is Q_{1,0}.
    Hence, if level 2 and 3 are included we want Q_{3,2}, which is at Q_l_l1[2]
    '''
    '''First determine jumps in levels'''
    jumps = np.where(diff_np(L_set)>1)[0] #index for jumps in terms of correlated results
    Q_temp = Q_d[L_set[1:]-1]; V_temp = V_d[L_set[1:]-1]; C_temp = C_d[L_set[1:]-1]
    '''No results are available for non-adjecant levels. So they are set to 0'''
    Q_temp[jumps] = 0; V_temp[jumps] = 0; C_temp[jumps] = 0;
    '''Insert in output variables'''
    Q_out[1:] = Q_temp; V_out[1:] = V_temp; C_out[1:] = C_temp; #Values for other levels
    '''Set number of paths for each level'''
    N_out = N*np.ones(len(L_set),dtype=np.int64)*(C_out > 0)
    return L_set,N_out,Q_out,V_out,C_out



# @njit(nogil=True)
def diff_np(a):
    '''Replacement for np.diff'''
    return a[1:]-a[:-1]

@njit(nogil=True,parallel=True)
def update_path_old(I,E,SS,C,N,N_diff,levels,Q,t0,T,mu,sigma,M,R,SC,R_anti,dR,boundary):
    for j in prange(len(I)):
        i=I[j]
        dt_f = (T-t0)/2**levels[i]
        x0,v0,v_l1_next = Q(N_diff[i])
        if i!=0:
            dt_c = (T-t0)/2**levels[i-1]
            with objmode(start1 = 'f8'):
                start1 = time.perf_counter()
            x_f,x_c = correlated(dt_f,dt_c,x0,v0,v_l1_next,t0,T,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR,boundary=boundary)
            with objmode(end1 = 'f8'):
                end1 = time.perf_counter()
            C_temp = (end1-start1)/N_diff[i]
            E_temp = np.mean(x_f-x_c)
            SS_temp = np.sum((x_f-x_c-E_temp)**2)
        else:
            # e = np.random.exponential(scale=1,size=N_diff[i]) #Could maybe be implemented in KDMC
            # tau = SC(x0,v0,e) #Could maybe be implemented in KDMC
            with objmode(start2 = 'f8'):
                start2 = time.perf_counter()
            x = KDMC(dt_f,x0,v0,t0,T,mu,sigma,M,R,SC,dR=dR,boundary=boundary)
            with objmode(end2 = 'f8'):
                end2 = time.perf_counter()
            C_temp = (end2-start2)/N_diff[i]
            E_temp = np.mean(x)
            SS_temp = np.sum((x-E_temp)**2)
        Delta = delta(E[i],E_temp,N[i],N_diff[i])
        E[i] = x_hat(N[i],N_diff[i],E[i],E_temp,Delta)
        SS[i] = Sfunc(N[i],N_diff[i],SS[i],SS_temp,Delta)
        '''Update cost like updating an average'''
        C[i] = x_hat(N[i],N_diff[i],C[i],C_temp,delta(C[i],C_temp,N[i],N_diff[i]))
        N[i] = N[i] + N_diff[i]
    return E,SS,N,N_diff,C

# @njit(nogil=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)#.astype(np.int64)


# @njit(nogil=True,parallel=True)
def update_path(I,E,SS,C,N,N_diff,levels,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary,update = True):
    cores = 1# if update else 8
    n = np.maximum(2,rnd1(N_diff/cores,0,np.empty_like(N_diff/cores)).astype(np.int64))
    # n = np.ones(2,dtype=np.int64)*2#np.maximum(np.ones(N_diff.size)*2,np.round(N_diff/cores,0,np.empty_like(N_diff/cores))).astype(np.int64)
    # for i,Nd in enumerate(N_diff):
    #     n[i] = max(2,round(Nd/cores))
    for j in range(len(I)):
        i=I[j]
        dt_f = (T-t0)/2**levels[i]
        C1=0;C2=0;
        for k in prange(cores):
            if i!=0:
                dt_c = (T-t0)/2**levels[i-1]
                with objmode(start1 = 'f8'):
                    start1 = time.perf_counter()
                x0,v0,v_l1_next = Q(n[i])
                x_f,x_c = correlated(dt_f,dt_c,x0,v0,v_l1_next,t0,T,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR,boundary=boundary)
                with objmode(end1 = 'f8'):
                    end1 = time.perf_counter()
                est_f = F(x_f);est_c=F(x_c)
                C_temp = (end1-start1)/n[i]
                E_temp = np.mean(est_f-est_c)
                SS_temp = np.sum((est_f-est_c-E_temp)**2)
            else:
                with objmode(start2 = 'f8'):
                    start2 = time.perf_counter()
                x = KDMC(dt_f,n[i],Q,t0,T,mu,sigma,M,R,SC,dR=dR,boundary=boundary)
                with objmode(end2 = 'f8'):
                    end2 = time.perf_counter()
                C_temp = (end2-start2)/n[i]
                est = F(x)
                E_temp = np.mean(est)
                SS_temp = np.sum((est-E_temp)**2)
            Delta = delta(E[i],E_temp,N[i],n[i])
            E[i] = x_hat(N[i],n[i],E[i],E_temp,Delta)
            if update:
                SS[i] = Sfunc(N[i],n[i],SS[i],SS_temp,Delta)
                '''Update cost like updating an average'''
                C[i] = x_hat(N[i],n[i],C[i],C_temp,delta(C[i],C_temp,N[i],n[i]))
            N[i] = N[i] + n[i]
    return E,SS,N,C

def lin_fit(x,y):
    p=1; n = x.size
    X = np.stack((np.ones(n),x),axis=1)
    normal_matrix = (X.T).dot(X)
    moment_matrix = (X.T).dot(y)
    return np.linalg.inv(normal_matrix).dot(moment_matrix)


# @njit(nogil=True)
def ml(e2,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=None,dR=None,tau=None,L=14,N_warm = 100,boundary=None,alpha = None,beta=None,gamma=None,levels=None):
    '''First do warm-up and select levels with L being the maximum level'''
    if levels is not None:
        L_num = len(levels)
        N = np.zeros(L_num,dtype=np.int64)
        E = np.zeros(L_num);V = np.zeros(L_num); C = np.zeros(L_num)
    else:
        levels,N,E,V,C = select_levels(L,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,N_warm,tau)
        '''Number of levels to use'''
        L_num = len(levels)
    L_start = L_num

    '''Variances will be updated and saved as sum of squares'''
    SS = np.maximum(0,(N-1)*V)
    '''Paths still needed to minimize total cost based on current information on variance and cost for each level'''
    N_diff = np.ones(L_num,dtype=np.int64)*N_warm - N
    # print(f'N_diff: {N_diff}')
    '''While loop to continue until RMSE fits with e2'''
    while True:
        '''Update paths based on N_diff'''
        while np.max(N_diff)>0:
            # print('update paths')
            I = np.where(N_diff > 0)[0] #Index for Levels that need more paths
            N_diff = np.minimum(N_diff,np.ones(len(N_diff),dtype=np.int64)*400_000)
            update = np.sum(N)==0 or (beta is None and gamma is None)
            E,SS,N,C = update_path(I,E,SS,C,N,N_diff,levels,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary,update=update)
            if update:V = SS/(N-1) #Update variance
            '''Determine number of paths needed with new information'''
            N_diff = np.maximum(0,np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64) - N)
            if L_num > L_start and alpha is not None and False:
                '''Find problematic/erratic bias values and require more paths for levels
                with eratic bias values. Deprecated part of function'''
                # index_bias = np.where((np.abs(E[L_start-1:-1])/np.abs(E[L_start:])< 2**(alpha/2)))[0] + 1
                index_bias = np.where((np.abs(E[L_start-1:-1])/np.abs(E[L_start:])< 1))[0] + 1
                index_bias = np.unique(np.append(index_bias,index_bias+1))
                print(index_bias)
                print(E)
                # print((np.abs(E[1:-1])/np.abs(E[2:])))
                N_diff[index_bias] += rnd1(N[index_bias]*0.1,0,np.empty_like(N[index_bias]*0.1)).astype(np.int64)
        '''Test bias is below e2/2'''
        if E.size>=4:
            test = np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2)
            '''Find jumps in levels. For those jumps, it is not the case that dt_c=2*dt_f
            Need to account for that when extrapolating values from previous levels.
            '''
        #     jumps_index = np.where(np.diff(levels)!=1)[0]+1
        #     if jumps_index.size==0: jumps_index= np.array([0],dtype=np.int64)
        #     jumps = levels[jumps_index]/levels[jumps_index-1]
        #     if alpha is None:
        #         L1 = max(1,np.where(1/2**levels<=1/R(1))[0][0])
        #         pa = lin_fit(np.arange(L1,L_num),np.log2(np.abs(E[L1:L_num]))); alpha = -pa[0]
        #     # M_t = max(2,round(dt_c/dt_f))
        #
        #     if np.max(jumps_index)<E.size-3:
        #         test = np.max(np.abs(E[-3:]))/(2**alpha-1) < np.sqrt(e2/2)
        #     else:
        #         count=0
        #         temp = 0
        #         for i in range(L_num-4,L_num):
        #             if i in jumps_index:
        #                 temp = max(np.abs(E[i])/(jumps[count]**alpha-1),temp)
        #                 count+=1
        #             else:
        #                 temp = max(np.abs(E[i])/(2**alpha-1),temp)
        else:
            test=False
        # test = max(abs(0.5*E[L_num-2]),abs(E[L_num-1])) < np.sqrt(e2/2)
        if test:
            break
        L_num += 1;
        levels = np.append(levels,levels[-1]+1)
        if levels[-1]==17:
            print('WARNING: maximum capacity reached!')
            break
        print('New level')
        if beta is None and gamma is None:
            N_diff = np.append(N_diff,N_warm).astype(np.int64)
            N = np.append(N,0).astype(np.int64)
            E = np.append(E,0.0); V = np.append(V,0.0); C = np.append(C,0.0); SS = np.append(SS,0.0)
        else:
            #Calculate constant from Theorem
            c2 = V[-1]*2**(levels[-2]*beta)
            V = np.append(V,c2*2**(-levels[-1]*beta))
            c3 = C[-1]*2**(levels[-2]*gamma)
            C = np.append(C,c3*2**(levels[-1]*gamma))
            E = np.append(E,0.0);
            N = np.append(N,0).astype(np.int64)
            N_diff = np.maximum(0,np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64) - N)
    return E,V,C,N,levels



@njit(nogil=True,parallel=True)
def convergence_tests(N,dt_list,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary):
    '''Calculates values for consistency test for each level given by dt_list'''
    cores = 8 #Controls parrelisation

    L = dt_list.size
    # sys.exit()
    b = np.zeros(L) #mean(F(x^f)-F(X^c))
    b2 = np.zeros(L) #mean((F(x^f)-F(X^c))^2)
    b3 = np.zeros(L) #mean((F(x^f)-F(X^c))^3)
    b4 = np.zeros(L) #mean((F(x^f)-F(X^c))^4)
    v = np.zeros(L) #mean(F(X))
    v2 = np.zeros(L) #mean(F(X)^2)
    var1 = np.zeros(L) #var(F(X))
    var2 = np.zeros(L) #var(F(X^f)-F(X^c))
    # kur1 = np.empty(L-1) #kurtosis calculated for biases
    cons = np.zeros(L) #consistency calculated for all biases
    cost1 = np.zeros(L) #cost for each level
    cost2 = np.zeros(L) #cost for each level bias



    if N%cores!=0:
        print('WARNING -  Number of samples is not divisble by 8!\n Please change N for optimal functionality')
    n = round(N/cores)
    for l in range(L):
        print(l)
        diff = np.empty((cores,n))
        val = np.empty((cores,n))
        for j in prange(cores):
            x0,v0,v_l1_next = Q(n)
            if l<L-1:
                # print(f'l={l}')
                with objmode(start1 = 'f8'):
                    start1 = time.perf_counter()
                x_f,x_c = correlated(dt_list[l+1],dt_list[l],x0,v0,v_l1_next,t0,T,mu,sigma,M,R,SC,R_anti=R_anti,dR=dR,boundary=boundary)
                with objmode(end1 = 'f8'):
                    end1 = time.perf_counter()
                cost2[l+1] += (end1-start1)
                diff[j,:] = F(x_f)-F(x_c)
            with objmode(start2 = 'f8'):
                start2 = time.perf_counter()
            x = KDMC(dt_list[l],n,Q,t0,T,mu,sigma,M,R,SC,dR=dR,boundary=boundary)
            with objmode(end2 = 'f8'):
                end2 = time.perf_counter()
            cost1[l] += (end2-start2)
            val[j,:] = F(x)
        cost1[l] = cost1[l]/(n*cores)
        v[l] = np.mean(val)
        v2[l] = np.mean(val**2)
        var1[l]  = v2[l] - v[l]**2
        if l<L-1:
            b[l+1] = np.mean(diff)
            b2[l+1] = np.mean(diff**2)
            b3[l+1] = np.mean(diff**3)
            b4[l+1] = np.mean(diff**4)
            var2[l+1] = b2[l+1]-b[l+1]**2
            cost2[l+1] = cost2[l+1]/(n*cores)
            # cons[l+1] = np.abs(b[l+1]+v[l]-v[l+1])/(3*(np.sqrt(var2[l+1])+ np.sqrt(var1[l])+np.sqrt(var1[l+1]))/np.sqrt(N))
    cons[1:] = np.abs(b[1:]+v[:-1]-v[1:])/(3*(np.sqrt(var2[1:])+np.sqrt(var1[:-1])+np.sqrt(var1[1:]))/np.sqrt(N))
    kur1 = np.append(0,(b4[1:]-4*b3[1:]*b[1:]+6*b2[1:]*b[1:]**2-3*b[1:]**4)/(b2[1:]-b[1:]**2)**2)
    return b,b2,b3,b4,v,v2,var1,var2,kur1,cons,cost1,cost2

jit_module(nopython=True,nogil=True)

def ml_test(N,N0,dt_list,E2,eps,Q,t0,T,mu,sigma,M,R,SC,F,logfile=None,R_anti=None,dR=None,tau=None,boundary=None,convergence=True,complexity=True):
    ''''
    filename for logfile should always begin with 'logfile_KD'


    '''
    save_file = logfile is not None
    L = dt_list.size
    if save_file:
        if logfile.name[0:10]!='logfile_KD':
            sys.exit(f'ERROR: name of logfile should start with "logfile_KD. Following argument was given: {logfile.name}"')
        now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        logfile.write("\n")
        logfile.write("*********************************************************\n")
        logfile.write(f"***Python ml_test for KD method on {now}         ***\n")
        logfile.write("\n")
        logfile.write("*********************************************************\n")
        logfile.write("*** Experiemnt setup  ***\n")
        logfile.write("*** S(x,v) = 1/sqrt(2*pi)*v^2*e^{-v^2/2}*(1+cos(2*pi*(x+1/2))) ***\n")
        logfile.write(f"*** R(x) = 1/eps^2 (ax+b), with a={eps**2*(R(1)-R(0))}, b={R(0)*eps**2} and eps = {eps}  ***\n")
        if 2!=F(np.array([2])):
            logfile.write(f"*** OBS!! replace this text with specification of quantity of interest F(3) = {F(np.array([3]))} ***\n")
        else:
            logfile.write("*** Quantity of interest is F(X) = X  ***\n")
        if boundary is None:
            logfile.write("*** No boundary conditions  ***\n")
        else:
            logfile.write("*** OBS!! replace this text with specification of boundary conditions  ***\n")
        logfile.write("*********************************************************\n")
        logfile.write("Convergence tests, kurtosis, telescoping sum check \n")
        logfile.write(f"*** using {N} samples and {L} levels ***\n")
        logfile.write(" l dt^f mean(F(X^f)-F(X^c)) mean(F(X^c))  var(F(X^f)-F(X^c)) var(F(X^c))")
        logfile.write(" cost(F(X^f)-F(X^c)) cost(F(X)) kurtosis consistency \n")
        logfile.write("---------------------------------------------------------\n")

    if convergence:
        print('Convergence test')
        b,b2,b3,b4,v,v2,var1,var2,kur1,cons,cost1,cost2 = convergence_tests(N,dt_list,Q,t0,T,mu,sigma,M,R,SC,F,R_anti,dR,boundary)
        # df = pd.DataFrame({'E(F(X^f)-F(X^c))': b,'E(F(X))': v,'var(F(X^f)-F(X^c))': var2,'var(F(X))':var1,'Kurtosis': kur1,'consistency check': cons,'cost(F(X^f)-F(X^c))':cost2,'cost(F(X))':cost1})
        # pd.set_option('max_columns',None)
        # print(df)
        # print(f'E(F(X^f)-F(X^c)) = {b}')
        # print(f'E(F(X)) = {v}')
        # print(f'var(F(X^f)-F(X^c)) = {var2}')
        # print(f'var(F(X)) = {var1}')
        # print(f'Kurtosis = {kur1}')
        # print(f'consistency check = {cons}')
        # print(f'cost(F(X^f)-F(X^c)) = {cost2}')
        # print(f'cost(F(X)) = {cost1}')
        # Linear regression to estimate alpha, beta and gamma. Only test for dt << eps^2
        if save_file:
            np.savetxt('resultfile'+logfile.name[7:],(dt_list,v,b,var1,var2,cost1,cost2,kur1,cons))
            logfile.write('\n*********************************************************\n')
            logfile.write('\n*** Linear regression estimates of MLMC paramters ***\n')
            logfile.write(f'\n*** regression is done for levels with dt << eps^2 = {eps**2} ***\n')
            logfile.write('*********************************************************\n')
        L1 = np.where(dt_list<1/R(0))[0][1]
        pa = np.polyfit(range(L1,L),np.log2(np.abs(b[L1:L])),1); alpha = -pa[0]
        pb = np.polyfit(range(L1,L),np.log2(np.abs(var2[L1:L])),1); beta = -pb[0]
        pg = np.polyfit(range(L1,L),np.log2(np.abs(cost2[L1:L])),1); gamma = pg[0]
        print(f'alpha= {alpha}, beta = {beta}, gamma= {gamma}')
        if save_file:
            logfile.write(f'alpha = {alpha} (exponent for weak convergence) \n')
            logfile.write(f'beta = {beta} (exponent for variance of bias estimate) \n')
            logfile.write(f'gamma = {gamma} (exponent for cost of bias estimate) \n')
            for i in range(dt_list.size):
                logfile.write(f'{i} {dt_list[i]} {b[i]} {v[i]} {var2[i]} {var1[i]} {cost2[i]} {cost1[i]} {kur1[i]} {cons[i]}\n')


    if complexity:
        ''''Levels are selected based on convergence results to exclude any warm-up
        procedure from the result of the complexity analysis'''
        print('Complexity tests')
        if save_file:
            logfile.write("\n*********************************************************\n")
            logfile.write("*** MLMC complexity test ***\n")
            logfile.write("*********************************************************\n")
            logfile.write(" e2 value mlmc_cost N_l dt \n")
            logfile.write("---------------------------------------------------------\n")
        if convergence:
            levels = select_levels_data(var1,var2[1:],t0,T)
            print(f'Levels selected: {levels}')
        else:
            levels=None

        data = {}
        pd.set_option('max_columns',None)
        # levels = np.array([0,1,12,13]) One coarse level
        levels = np.array([6,7],dtype=np.int64)
        # levels = np.array([3,4],dtype=np.int64)
        # levels=None
        if convergence:
            '''Find index where they transition'''
            i = 0
            old = 0
            for va in var2[1:]:
                if va < old:
                    break
                i+=1
                old=va
            # print(f'slope of variance before:{np.polyfit(range(1,i-2),np.log2(np.abs(var2[1:i-2])),1)}')
            # print(f'slope of variance after:{np.polyfit(range(i+2,var2.size),np.log2(np.abs(var2[i+2:])),1)}')
            # print(f'slope of bias before:{np.polyfit(range(1,i-2),np.log2(np.abs(bias[1:i-2])),1)}')
            alpha = -np.polyfit(range(i,var2.size),np.log2(np.abs(b[i:])),1)[0]
            # print(f'slope of bias after:{alpha}')
            # print(f'slope of cost:{np.polyfit(range(i,cost2.size),np.log2(np.abs(cost2[i:])),1)}')
            '''Complexity test based on estimations in convergence test'''
            dfs = {}
            pd.set_option('display.float_format', '{:.2E}'.format)
            for e2 in E2:
                L=4 + i
                E = np.append(v[i],b[i:L])
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
                    # print(df)
                    dfs[e2] = df
            # print(dfs)
            print(L)
            '''Plotting number of paths and computational costs as functions of MSE'''
            fig, (ax1, ax2) = plt.subplots(1, 2)
            N = []
            Cost = []
            end = len(dfs)
            start = end-min(end,6)
            for e2 in E2[start:end]:
                N += [dfs[e2]['N'][:-1]]
                Cost += [dfs[e2]['N C'][N[-1].size]]
                # print(Cost)
                ax1.plot(range(N[-1].size),N[-1],label=r'$E^2 = ${:.2E}'.format(e2))
            ax2.loglog(E2[start:end],Cost,label='KD')
            c4 = np.mean(Cost*E2[start:end])
            c4 *= 2
            ax2.plot(E2[start:end],c4*1/E2[start:end],label= r'$\mathcal{O}(E^{-2})$')
            ax1.set_yscale('log')
            ax2.set_xlabel(r'$E^2$')
            ax2.set_ylabel(r'Total costs')
            ax1.set_xlabel(r'Levels')
            ax1.set_ylabel(r'Paths')
            ax2.legend()
            ax1.legend()
            if save_file:
                filename = f'complexity_results'+logfile.name[7:]
                plt.savefig(filename)

        else:
            # Projected cost based on Multilevel Theorem
            cost_e=0; l = 1
            e2 = 1/2**l
            WC = []
            print('Compile')
            ml(e2,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=R_anti,dR=dR,tau=tau,L=14,N_warm=N0,boundary=boundary,alpha=1.15,beta=3.0,gamma=0.0,levels=levels)
            print('Compilation done')
            while cost_e<3000:
                print(f'MSE= {e2}')
                start = time.perf_counter()
                E,V,C,N,levels_out = ml(e2,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=R_anti,dR=dR,tau=tau,L=14,N_warm=N0,boundary=boundary,alpha=1.30,levels=levels)
                WC +=[time.perf_counter()-start]
                data[e2] = {'dt':(T-t0)/2**levels_out,'N_l':N,'E':E,'V_l':V,'V[E]':V/N,'C_l':C,'N_l C_l':N*C}
                df = pd.DataFrame(data[e2])
                df = df.append(pd.DataFrame({'dt':' ','N_l':' ','E':[np.sum([e for e in data[e2]['E']])],'V_l':' ','V[E]':[np.sum([v for v in data[e2]['V[E]']])],'C_l':[np.sum([c for c in data[e2]['C_l']])],'N_l C_l':[np.sum(n*c for n,c in zip(data[e2]['N_l'],data[e2]['C_l']))]}))
                print(df)
                if save_file:
                    name = f'resultfile_complexity_{e2}'+logfile.name[7:]
                    df.to_csv(name,index=False)
                    logfile.write(f" {e2} {np.sum(E)} {np.dot(C,N)} {N} {(T-t0)/2**levels_out} \n")
                #Actual simulation cost
                cost_s = WC[-1]#np.dot(C,N)
                #Constant from Theorem
                c4 = cost_s*e2
                l+=1
                e2 = 1/2**l
                cost_e = c4/e2

            if save_file:
                name = f'wall_clock_time_{e2}'+logfile.name[7:]
                np.savetxt(name,WC)
        # for e2 in E2:
        #     print(f'MSE: {e2}')
        #     start = time.time()
        #     E,V,C,N,levels_out = ml(e2,Q,t0,T,mu,sigma,M,R,SC,F,R_anti=R_anti,dR=dR,tau=tau,L=14,N_warm=N0,boundary=boundary,alpha=1.15,levels=levels)
        #     print(f'Time: {time.time()-start}')
        #     data[e2] = {'dt':(T-t0)/2**levels_out,'N_l':N,'E':E,'V_l':V,'V[E]':V/N,'C_l':C,'N_l C_l':N*C}
        #     df = pd.DataFrame(data[e2])
        #     df = df.append(pd.DataFrame({'dt':' ','N_l':' ','E':[np.sum([e for e in data[e2]['E']])],'V_l':' ','V[E]':[np.sum([v for v in data[e2]['V[E]']])],'C_l':[np.sum([c for c in data[e2]['C_l']])],'N_l C_l':[np.sum(n*c for n,c in zip(data[e2]['N_l'],data[e2]['C_l']))]}))
        #     print(df)
        #     if save_file:
        #         name = f'resultfile_complexity_{e2}'+logfile.name[7:]
        #         df.to_csv(name,index=False)
        #         logfile.write(f" {e2} {np.sum(E)} {np.dot(C,N)} {N} {(T-t0)/2**levels_out} \n")



    if save_file:
        logfile.write('\n')
        logfile.close()

    if convergence:
        plt.plot(dt_list[1:],var2[1:],':',label='var(F(x^f)-F(X^c))')
        plt.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
        plt.plot(dt_list[1:],np.abs(b[1:]),':',label='mean(|F(x^f)-F(X^c)|)')
        plt.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
        # plt.title(f'Plot of variance and bias')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        if save_file:plt.savefig(f'var_bias'+logfile.name[7:])
        plt.figure()
        plt.plot(range(1,L),kur1[1:],':',label='kurtosis')
        plt.title(f'Plot of kurtosis')
        if save_file:plt.savefig(f'kurtosis'+logfile.name[7:])
        plt.figure()
        plt.plot(range(1,L),cons[1:],':',label='kurtosis')
        plt.title(f'Plot check of consistency')
        if save_file:plt.savefig(f'consistency_check'+logfile.name[7:])
        if not save_file:plt.show()
