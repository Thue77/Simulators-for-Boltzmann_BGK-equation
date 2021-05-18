import numpy as np
from .correlated import correlated,correlated_ts
from .mc import mc
from .AddPaths import delta,x_hat,Sfunc
import time
import pandas as pd
from numba import njit,jit_module,prange,objmode
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

@njit(nogil=True)
def select_levels(t0,T,M_t,eps,r,F,strategy=1,cold_start=True,N=100,boundary=None):
    '''
    strategy: indicates if strategy 1 or 2 is used
    cold_start: indicates if the variance and estimates are calculated for an
    initial number of paths
    '''
    if strategy==1:
        dt_0 = np.minimum((eps)**2/r(np.array([0])),T-t0)
        levels = dt_0/M_t**np.arange(0,4)#np.array([dt_0,dt_0/M_t])
        N_out=np.zeros(4,dtype=np.int64)
        N_diff=np.ones(4,dtype=np.int64)*N
        SS_out = np.zeros(4);C_out = np.zeros(4); E_out=np.zeros(4)
    else:
        dt_1 = np.minimum((eps/10)**2,0.025)
        levels = np.array([float(T-t0),dt_1,dt_1/M_t])
        N_out=np.zeros(3,dtype=np.int64)
        N_diff=np.ones(3,dtype=np.int64)*N
        SS_out = np.zeros(3);C_out = np.zeros(3); E_out=np.zeros(3)
    return levels,N_out,N_diff,E_out,SS_out,C_out


def rnd1(x, decimals, out):
    return np.round_(x, decimals, out).astype(np.int64)

@njit(nogil=True,parallel=True)
def update_paths(I,E,SS,C,N,N_diff,levels,t0,T,M_t,eps,Q,M,r,F,boundary,strategy=1,rev=False,diff=False):
    # levels = ll.copy()
    cores = 8
    n = np.maximum(2,rnd1(N_diff/cores,0,np.empty_like(N_diff)).astype(np.int64))
    for j in range(len(I)):
        i=I[j]
        # print(i)
        dt_f = levels[i]
        # x0,v0,v_l1_next = Q(N_diff[i])
        for k in prange(cores):
            if i!=0:
                if strategy==1 or i!=1:
                    with objmode(start1 = 'f8'):
                        start1 = time.perf_counter()
                    if rev:
                        x_f,x_c = correlated_ts(dt_f,M_t,t0,T,eps,n[i],Q,M,r,boundary=boundary,strategy=strategy,diff=diff)
                    else:
                        x_f,x_c = correlated(dt_f,M_t,t0,T,eps,n[i],Q,M,r,boundary=boundary,strategy=strategy,diff=diff)
                    with objmode(end1 = 'f8'):
                        end1 = time.perf_counter()
                else:
                    with objmode(start1 = 'f8'):
                        start1 = time.perf_counter()
                    x_f,x_c = correlated(dt_f,round(levels[0]/dt_f),t0,T,eps,n[i],Q,M,r,boundary=boundary,strategy=strategy,diff=diff)
                    with objmode(end1 = 'f8'):
                        end1 = time.perf_counter()
                est_f = F(x_f);est_c=F(x_c)
                C_temp = (end1-start1)/n[i]
                E_temp = np.mean(est_f-est_c)
                SS_temp = np.sum((est_f-est_c-E_temp)**2)
            else:
                with objmode(start2 = 'f8'):
                    start2 = time.perf_counter()
                x = mc(dt_f,t0,T,n[i],eps,Q,M,r,boundary=boundary,rev=rev,diff=diff)
                with objmode(end2 = 'f8'):
                    end2 = time.perf_counter()
                est = F(x)
                C_temp = (end2-start2)/n[i]
                E_temp = np.mean(est)
                SS_temp = np.sum((est-E_temp)**2)
            Delta = delta(E[i],E_temp,N[i],n[i])
            E[i] = x_hat(N[i],n[i],E[i],E_temp,Delta)
            SS[i] = Sfunc(N[i],n[i],SS[i],SS_temp,Delta)
            '''Update cost like updating an average'''
            C[i] = x_hat(N[i],n[i],C[i],C_temp,delta(C[i],C_temp,N[i],n[i]))
            N[i] = N[i] + n[i]
    return E,SS,N,C


# @njit(nogil=True)
def ml(e2,Q,t0,T,M_t,eps,M,r,F,N_warm=40,boundary=None,strategy=1,alpha=None,beta=None,gamma=None,rev=False,diff=False):
    '''
    e2: bound on mean square error
    Q: initial distribution
    t0: starting time
    T: end time
    M_t: dt_c/dt_f, where dt_c is coarse step size and dt_f is fine step size
    eps: mean free path, i.e. epsilon in model
    M: equilibrium distribution for velocity
    r: collision rate
    F: function used to find quantity of interest, E(F(X,V)).
    '''
    levels,N,N_diff,E,SS,C = select_levels(t0,T,M_t,eps,r,F,N=N_warm,strategy=strategy)
    '''
    levels: a list of step sizes for each level
    N: list of number of paths used
    N_diff: list of number of paths needed at every level
    E: estimation of quantity of interest at each level
    SS: sum of squares at each level
    C: Cost at each level
    '''
    L = len(levels)
    # C = M_t**np.arange(0,L,1); C[1:] = C[1:] + M_t**np.arange(0,L-1,1)
    '''While loop to continue until RMSE fits with e2'''
    while True:
        '''Update paths based on N_diff'''
        while np.max(N_diff)>0:
            I = np.where(N_diff > 0)[0] #Index for Levels that need more paths
            N_diff = np.minimum(N_diff,np.ones(len(N_diff),dtype=np.int64)*8_000_000)
            # print(f'index where more paths are needed: {I}, N_diff: {N_diff}')
            E,SS,N,C = update_paths(I,E,SS,C,N,N_diff,levels,t0,T,M_t,eps,Q,M,r,F,boundary,strategy,rev=rev,diff=diff)
            V = SS/(N-1) #Update variance
            '''Determine number of paths needed with new information'''
            N_diff = np.ceil(2/e2*np.sqrt(V/C)*np.sum(np.sqrt(V*C))).astype(np.int64) - N
        '''Test bias is below e2/2 - to do so it is necessary to know exponent
        in weak error'''
        # test = max(abs(0.5*E[L-2]),abs(E[L-1])) < np.sqrt(e2/2)
        if E.size>=4:
            if alpha is None:
                L1 = max(1,np.where(levels<=eps**2)[0][0])
                pa = lin_fit(np.arange(L1,L),np.log2(np.abs(E[L1:L]))); alpha = -pa[0]
            test = np.max(np.abs(E[-3:]))/(M_t**alpha-1) < np.sqrt(e2/2)
            # print('Extrapolated values for bias:')
            # print(np.abs(E[-3:])/M_t**(np.flip(np.arange(0,3))*alpha))
        else:
            test=False
        if test:
            break
        L += 1;
        if L==18:
            print('WARNING: maximum capacity reached!')
            break
        print('Level added')
        print(L)
        # print(f'New level: {L}')
        N_diff = np.append(N_diff,N_warm).astype(np.int64)
        N = np.append(N,0).astype(np.int64)
        E = np.append(E,0.0); V = np.append(V,0.0); SS = np.append(SS,0.0)
        C = np.append(C,0.0);#np.append(C,M_t**(L-1)+M_t**(L-2)); #
        if strategy==1:
            levels = np.append(levels,levels[0]/(M_t**(L-1)))
        else:
            levels = np.append(levels,levels[1]/(M_t**(L-2)))
    return E,V,C,N,levels


def lin_fit(x,y):
    p=1; n = x.size
    X = np.stack((np.ones(n),x),axis=1)
    normal_matrix = (X.T).dot(X)
    moment_matrix = (X.T).dot(y)
    return np.linalg.inv(normal_matrix).dot(moment_matrix)



@njit(nogil=True,parallel=True)
def convergence_tests(N,dt_list,Q,t0,T,M_t,eps,M,r,F,boundary,strategy,rev=False,diff=False):
    '''Calculates values for consistency test for each level given by dt_list'''
    cores = 8 #Controls parrelisation

    L = dt_list.size
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
    # all_diff = np.zeros((N,L))

    for l in range(L):
        diff_vec = np.empty((cores,n))
        val = np.empty((cores,n))
        for j in prange(cores):
            if l<L-1:
                # print(f'l={l}')
                with objmode(start1 = 'f8'):
                    start1 = time.perf_counter()
                if rev:
                    x_f,x_c = correlated_ts(dt_list[l+1],M_t,t0,T,eps,n,Q,M,r,boundary=boundary,diff=diff)
                else:
                    x_f,x_c = correlated(dt_list[l+1],M_t,t0,T,eps,n,Q,M,r,boundary=boundary,diff=diff)
                with objmode(end1 = 'f8'):
                    end1 = time.perf_counter()
                cost2[l+1] += (end1-start1)
                diff_vec[j,:] = F(x_f)-F(x_c)
            with objmode(start2 = 'f8'):
                start2 = time.perf_counter()
            x = mc(dt_list[l],t0,T,n,eps,Q,M,r,boundary=boundary,rev=rev,diff=diff)
            with objmode(end2 = 'f8'):
                end2 = time.perf_counter()
            cost1[l] += (end2-start2)
            val[j,:] = F(x)
        cost1[l] = cost1[l]/N
        v[l] = np.mean(val)
        v2[l] = np.mean(val**2)
        var1[l]  = v2[l] - v[l]**2
        if l<L-1:
            # all_diff[:,l+1] = diff.flatten()
            b[l+1] = np.mean(diff_vec)
            b2[l+1] = np.mean(diff_vec**2)
            b3[l+1] = np.mean(diff_vec**3)
            b4[l+1] = np.mean(diff_vec**4)
            var2[l+1] = b2[l+1]-b[l+1]**2
            cost2[l+1] = cost2[l+1]/N
            # cons[l+1] = np.abs(b[l+1]+v[l]-v[l+1])/(3*(np.sqrt(var2[l+1])+ np.sqrt(var1[l])+np.sqrt(var1[l+1]))/np.sqrt(N))
    cons[1:] = np.abs(b[1:]+v[:-1]-v[1:])/(3*(np.sqrt(var2[1:])+np.sqrt(var1[:-1])+np.sqrt(var1[1:]))/np.sqrt(N))
    kur1 = np.append(0,(b4[1:]-4*b3[1:]*b[1:]+6*b2[1:]*b[1:]**2-3*b[1:]**4)/(b2[1:]-b[1:]**2)**2)
    return b,b2,b3,b4,v,v2,var1,var2,kur1,cons,cost1,cost2#,all_diff


jit_module(nopython=True,nogil=True)




def ml_test(N,N0,dt_list,E2,Q,t0,T,M_t,eps,M,r,F,logfile,boundary=None,strategy=1,convergence=True,complexity=True,rev=False,diff=False):
    ''''
    filename for logfile should always begin with 'logfile_APS'


    '''
    save_file= logfile is not None
    L = dt_list.size
    if save_file:
        if logfile.name[0:11]!='logfile_APS':
            sys.exit('ERROR: name of logfile should start with "logfile_APS"')
        now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        logfile.write("\n")
        logfile.write("*********************************************************\n")
        logfile.write(f"***Python ml_test for APS method on {now}         ***\n")
        if rev:
            logfile.write(f"***    Reverse splitting is used!!      ***\n")
        if diff:
            logfile.write(f"***    Altered diffusive coefficient is used!!      ***\n")
        logfile.write("\n")
        logfile.write("*********************************************************\n")
        logfile.write("*** Experiemnt setup  ***\n")
        logfile.write("*** S(x,v) = 1/sqrt(2*pi)*v^2*e^{-v^2/2}*(1+cos(2*pi*(x+1/2))) ***\n")
        logfile.write(f"*** r(x) = ax+b, with a={(r(np.array([1]))-r(np.array([0])))}, b={r(np.array([0]))} and eps = {eps}  ***\n")
        if 2!=F(2):
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
        b,b2,b3,b4,v,v2,var1,var2,kur1,cons,cost1,cost2 = convergence_tests(N,dt_list,Q,t0,T,M_t,eps,M,r,F,boundary=boundary,strategy=strategy,rev=rev,diff=diff)
        print('Convergence test DONE')
        if save_file:
            for i in range(dt_list.size):
                logfile.write(f'{i} {dt_list[i]} {b[i]} {v[i]} {var2[i]} {var1[i]} {cost2[i]} {cost1[i]} {kur1[i]} {cons[i]}\n')

        # Linear regression to estimate alpha, beta and gamma. Only test for dt << eps^2
        L1 = np.where(dt_list<eps**2/(r(np.array([0]))))[0][1]
        pa = np.polyfit(range(L1,L),np.log2(np.abs(b[L1:L])),1); alpha = -pa[0]
        pb = np.polyfit(range(L1,L),np.log2(np.abs(var2[L1:L])),1); beta = -pb[0]
        pg = np.polyfit(range(L1,L),np.log2(np.abs(cost2[L1:L])),1); gamma = pg[0]
        print(f'alpha= {alpha}, beta = {beta}, gamma= {gamma}')
        if save_file:
            np.savetxt('resultfile'+logfile.name[7:],(dt_list,v,b,var1,var2,cost1,cost2,kur1,cons))
            logfile.write('\n*********************************************************\n')
            logfile.write('\n*** Linear regression estimates of MLMC paramters ***\n')
            logfile.write(f'\n*** regression is done for levels with dt << eps^2 = {eps**2} ***\n')
            logfile.write('*********************************************************\n')
            logfile.write(f'alpha = {alpha} (exponent for weak convergence) \n')
            logfile.write(f'beta = {beta} (exponent for variance of bias estimate) \n')
            logfile.write(f'gamma = {gamma} (exponent for cost of bias estimate) \n')


    if complexity:
        if save_file:
            logfile.write("\n*********************************************************\n")
            logfile.write("*** MLMC complexity test ***\n")
            logfile.write("*********************************************************\n")
            logfile.write(" e2 value mlmc_cost N_l dt \n")
            logfile.write("---------------------------------------------------------\n")
        data = {}
        for e2 in E2:
            print(f'MSE= {e2}')
            E,V,C,N,levels = ml(e2,Q,t0,T,M_t,eps,M,r,F,N0,boundary=boundary,strategy=strategy,alpha=alpha,rev=rev,diff=diff)
            if save_file:
                logfile.write(f" {e2} {np.sum(E)} {np.dot(C,N)} {N} {levels} \n")
            data[e2] = {'dt':levels,'N_l':N,'E':E,'V_l':V,'V[E]':V/N,'C_l':C,'N_l C_l':N*C}

        pd.set_option('max_columns',None)
        for e2,d in data.items():
            print(f'MSE: {e2}')
            df = pd.DataFrame(d)
            df = df.append(pd.DataFrame({'dt':' ','N_l':' ','E':[np.sum([e for e in d['E']])],'V_l':' ','V[E]':[np.sum([v for v in d['V[E]']])],'C_l':[np.sum([c for c in d['C_l']])],'N_l C_l':[np.sum(n*c for n,c in zip(d['N_l'],d['C_l']))]}))
            # df = df.append(pd.DataFrame({'dt':' ','N_l':' ','E':' ','V_l':' ','C_l':' ','N_l C_l':' '}))
            print(df)
            if save_file:
                name = f'resultfile_complexity_{e2}'+logfile.name[7:]
                df.to_csv(name,index=False)
    if save_file:
        logfile.write('\n')
        logfile.close()

    if convergence and False:
        plt.plot(dt_list[1:],var2[1:],':',label='var(F(x^f)-F(X^c))')
        plt.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
        plt.plot(dt_list[1:],np.abs(b[1:]),':',label='mean(|F(x^f)-F(X^c)|)')
        plt.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
        plt.title(f'Plot of variance and bias')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.figure()
        plt.plot(range(1,L),kur1[1:],':',label='kurtosis')
        plt.title(f'Plot of kurtosis')
        plt.figure()
        plt.plot(range(1,L),cons[1:],':',label='kurtosis')
        plt.title(f'Plot check of consistency')
        plt.show()
