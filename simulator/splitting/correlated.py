from .one_step import phi_APS,phi_APS_new
import matplotlib.pyplot as plt
import numpy as np
from numba import njit,jit_module
from numba import prange
import sys

@njit(nogil=True)
def correlated(dt_f,M_t,t,T,eps,N,Q,B,r,boundary=None,strategy = 1,diff=False):
    '''
    M_t: defined s.t. dt_c=M_t dt_f
    t: starting time
    eps: diffusive parameter
    N: number of paths
    Q: Initial distribution
    M: velocity distribution
    '''
    dt_c = dt_f*M_t
    first_level = np.abs(T-dt_c)<1e-7
    # if first_level:
    #     print(f'dt_f: {dt_f}, dt_c: {dt_c}, M_t: {M_t}')
    x_f,v_f,_ = Q(N)
    x_c = x_f.copy()
    v_bar_c = v_f.copy()
    if strategy == 3 and first_level:
        v_bar_all = np.zeros((N,M_t))
        # v_bar_all[0,:] = v_bar_c.copy()
    v_c = v_f.copy()
    while t<T:
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))
        for m in range(M_t):
            C = (U.T>=eps**2/(eps**2+dt_f*r(x_f))).T #Indicates if collisions happen
            x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r,boundary=boundary,diff=diff)
            v_bar_c[C[:,m]] = v_f[C[:,m]]
            if strategy == 3 and first_level:
                v_bar_all[:,m] = v_f
        if strategy == 3 and first_level:
            z_c = improved_corr(dt_f,M_t,eps,x_f,x_c,v_bar_all,v_c,Z,r,first_level)
        else:
            z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c,_ = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c,boundary=boundary,diff=diff)
        t += dt_c
    return x_f,x_c
@njit(nogil=True)
def improved_corr(dt_f,M_t,eps,x_f,x_c,v_all,v_c,z,r,first_level):
    '''Correlates brownian numbers based on article "Multilevel Monte Carlo with
    Improved Correlation for Kinetic Equations in Diffusive Scaling" '''
    p_nc = eps**2/(eps**2+dt_f*r(x_f)); p_c = 1-p_nc
    psi_W = 1/np.sqrt(M_t)*np.sum(z,axis=1)
    if not first_level:
        v_var = np.ones(len(x_f))*(M_t + 2*(p_nc*(p_nc**M_t+M_t*p_c-1))/p_c**2)
    else:
        v_var = np.ones(len(x_f))*(3*+M_t-4)#(3*+M_t-2**(2-M_t)-4)
    psi_T = 1/np.sqrt(v_var)*np.sum(v_all,axis=1)
    ''''Calculating weights'''
    if not first_level:
        dt_c = M_t*dt_f
        #diffusion coefficient for fine path
        D_l = v_all[:,-1]**2*dt_f/(eps**2+dt_f*r(x_f))#dt_f/(eps**2+dt_f*r(x_f))#
        #Diffusion coefficient for coarse path
        D_l1 = v_c**2*dt_c/(eps**2+dt_c*r(x_c))#dt_c/(eps**2+dt_c*r(x_c))#
        theta = D_l*(2*dt_c*D_l1+dt_c**2*(eps/(eps**2+dt_c*r(x_c)))**2)/(D_l1*(2*M_t*dt_f*D_l+dt_f**2*(eps/(eps**2+dt_f*r(x_f)))**2*(M_t+2*p_nc*(p_nc**M_t+M_t*p_c-1)/p_c**2)))
    else:
        theta = np.ones(len(x_f))*(4*M_t+1)/(7*M_t-4)
    return np.sqrt(theta)*psi_W + np.sqrt(1-theta)*psi_T


'''My correlation with inspiration from strang splitting'''
@njit(nogil=True)
def correlated_ts(dt_f,M_t,t,T,eps,N,Q,B,r,boundary=None,strategy = 1,diff=False):
    '''
    M_t: defined s.t. dt_c=M_t dt_f
    t: starting time
    eps: diffusive parameter
    N: number of paths
    Q: Initial distribution
    M: velocity distribution
    '''
    dt_c = dt_f*M_t
    first_level = np.abs(T-dt_c)<1e-7
    # if first_level:
    #     print(f'dt_f: {dt_f}, dt_c: {dt_c}, M_t: {M_t}')
    x_f,v_f,_ = Q(N)
    x_c = x_f.copy()
    v_bar_c = v_f.copy()
    v_c = v_f.copy()
    while t<T:
        v_bar_all = np.zeros((N,M_t))
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))
        for m in range(M_t):
            C = (U.T>=eps**2/(eps**2+dt_f*r(x_f))).T #Indicates if collisions happen
            x_f,v_f,v_bar_f = phi_APS_new(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r,boundary=boundary,diff=diff)
            v_bar_all[C[:,m],m] = v_f[C[:,m]]
        v_bar_c,z_c = cor_rv(M_t,Z,C,v_bar_all)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c,_ = phi_APS_new(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c,boundary=boundary,diff=diff)
        t += dt_c
    return x_f,x_c

@njit(nogil=True,parallel=True)
def cor_rv(M_t,Z,C,v_bar_all):
    z = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
    '''Determine influence of each v*'''
    '''The number of 1's in C_a indicate the number of steps that the first velocity
    influences and the number 2's indicate the number of steps that the second
    velocity influences'''
    C_a = np.zeros_like(v_bar_all)
    # print(C)
    n = C.shape[0]
    for j in prange(n):
        C_a[j,:] = np.cumsum(C[j,:])
    # C_a = np.cumsum(C,axis=1)
    # print(f'C_a= {C_a}')

    #number of steps affected by collisions
    steps = np.count_nonzero(C_a>0,axis=1)
    #number of steps that each collision affects. count[0,1]: number of steps affected by second collision for path 0
    temp = np.zeros_like(v_bar_all).astype(np.int64)
    # i_c = np.zeros(M_t,dtype=np.int64)
    for i in prange(M_t):
        temp[:,i] = np.count_nonzero(C_a==i+1,axis=1)
        # i_c = np.sum(count[:,0:i])

    #fit index of number of steps affected by collision with the index of the collision
    # start = np.minimum(M_t-np.sum(temp,axis=1),M_t-1).astype(np.int64)
    count = np.zeros_like(v_bar_all).astype(np.int64)
    put_np(count,temp,M_t)
    # for i in prange(M_t):
    #     count[range(n),start] = temp[range(n),i]
    #     start = np.minimum(start+temp[:,i],M_t-1).astype(np.int64)


    # print(f'count={count}, steps = {steps}')
    theta = np.zeros_like(v_bar_all)
    '''Cannot divide with steps for paths where no collisions occurs'''
    index = np.where(steps>0)[0]
    theta[index,:] = (count[index,:].T/steps[index]).T
    # for i in index:
    #     index_zero = theta[i,:]==0
    #     theta[i,index_zero] = 1
    # print(f'steps: {steps},\n count= {count},\n C_a: {C_a},\n theta: {theta}\n v_bar_all: {v_bar_all}')
    # print(f'output: {np.sum(np.sqrt(theta)*v_bar_all,axis=1)}')

    return np.sum(np.sqrt(theta)*v_bar_all,axis=1),z
@njit(nogil=True)
def put_np(count,temp,M_t):
    # print(f'temp: {temp}')
    for i in range(count.shape[0]):
        start = M_t-np.sum(temp[i,:])
        for j in range(M_t):
            if start>=M_t:
                break
            count[i,start] = temp[i,j]
            start = start + temp[i,j]






'''Function to make different plot tests for homogenous version of correlated method'''
def correlated_test(dt_f,M_t,t,T,eps,N,Q,B,r=1,plot=False,plot_var=False,rev = False,diff=False):
    '''
    M_t: defined s.t. dt_c=M_t dt_f
    t: starting time
    eps: diffusive parameter
    N: number of paths
    Q: Initial distribution
    M: velocity distribution
    '''
    dt_c = dt_f*M_t
    x_f,_,v_f = Q(N)
    # v_f = v_char(dt_f,eps)*v_bar_f; v_c = v_char(dt_c,eps)*v_bar_f
    x_c = x_f.copy()
    v_bar_c = v_f.copy()
    v_c = v_f.copy()
    if plot_var:
        var_d = [0]
        var_f = [0]
        var_c = [0]
    if plot:
        X_f = [x_f] if not rev else []
        X_c = [x_c] if not rev else []
        C1 = [(0.0,np.array([0.0]))] if not rev else []
        C2 = [(0.0,np.array([0.0]))] if not rev else []
    while t<T:
        if rev: v_bar_all = np.zeros((N,M_t))
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))

        C = (U>=eps**2/(eps**2+dt_f*r(x_f))) #Indicates if collisions happen
        # print(f'probability = {1-eps**2/(eps**2+dt_f*r)}')
        print(f'*********************** t = {t}*************************************')
        for m in range(M_t):
            if rev:
                if plot:
                    X_f += [x_f]
                    if C[:,m]: C1 += [(t+m*dt_f,x_f)]
                x_f,v_f,v_bar_f = phi_APS_new(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r,diff=diff)
                v_bar_all[C[:,m],m] = v_f[C[:,m]]
            else:
                x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r,diff=diff)
                v_bar_c[C[:,m]] = v_f[C[:,m]]#v_bar_f[C[:,m]]
            # print(f'v_last: {v_last}')
            if plot and not rev:
                X_f += [x_f]
                if C[:,m]: C1 += [(t+(m+1)*dt_f,x_f)]
        print('---------- Fine data ----------------')
        print(f'v_f: {v_f},\n v_bar_all: {v_bar_all},\n C: {C}')
        u_c = max_np(U,axis=1)**M_t
        if rev:
            v_bar_c,z_c = cor_rv(M_t,Z,C,v_bar_all)
            # print(f'Collisions: {C}')
            # print(f'v_f = {v_bar_all}, \n v_c = {v_c}')
            # sys.exit()
            if plot:
                X_c += [x_c]
                if u_c >=eps**2/(eps**2+dt_c*r(x_c)): C2 += [(t,x_c)]
            x_c,v_c,_ = phi_APS_new(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c,diff=diff)
        else:
            z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
            x_c,v_c,_ = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c,diff=diff)
        print('---------- Coarse data ----------------')
        print(f'v_f: {v_c},\n v_bar_c: {v_bar_c}  u_c: {u_c},\n C: {u_c>=eps**2/(eps**2+dt_c*r(x_c))}')
        t += dt_c
        if plot and not rev:
            X_c += [x_c]
            if u_c >=eps**2/(eps**2+dt_c*r(x_c)): C2 += [(t,x_c)]
        if plot_var:
            var_d += [np.var(x_f-x_c)]
            var_f += [np.var(x_f)]
            var_c += [np.var(x_c)]
    if rev and plot:
        X_f += [x_f]
        X_c += [x_c]
    if plot:
        plt.plot(np.arange(0,T+dt_f,dt_f),X_f,'.-') if not rev else plt.plot(np.arange(0,T+dt_f,dt_f),X_f,'.-')
        plt.plot(np.arange(0,T+dt_c,dt_c),X_c,'.-') if not rev else plt.plot(np.arange(0,T+dt_c,dt_c),X_c,'.-')
        plt.plot([t for t,_ in C1],[x[0] for _,x in C1],'x',color='blue')
        plt.plot([t for t,_ in C2],[x[0] for _,x in C2],'x',color='red')
        # print(f'C1: {C1} \n C2: {C2}')
        plt.grid(color='black', lw=1.0)
        plt.show()
    if plot_var:
        plt.plot(np.arange(0,11),var_d,'.-',color='green',label='Difference')
        plt.plot(np.arange(0,11),var_f,'.-',color='blue',label='Fine')
        plt.plot(np.arange(0,11),var_c,'.-',color='orange',label='Coarse')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.legend(title='Type of path')
        plt.show()
    return x_f,x_c


'''Heterogeneous version where r = r(x)'''
@njit(nogil=True)
def correlated_non_hom(dt_f,M_t,t,T,eps,N,Q,B,r,plot=False,plot_var=False):
    '''
    M_t: defined s.t. dt_c=M_t dt_f
    t: starting time
    eps: diffusive parameter
    N: number of paths
    Q: Initial distribution
    M: velocity distribution
    '''
    dt_c = dt_f*M_t
    x_f,_,v_f = Q(N)
    x_c = x_f.copy()
    v_bar_c = v_f.copy()
    v_c = v_f.copy()
    while t<T:
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))
        C = (U>=eps**2/(eps**2+dt_f*r)) #Indicates if collisions happen
        for m in range(M_t):
            x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r(x_f))
            v_bar_c[C[:,m]] = v_f[C[:,m]]
        z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c,_ = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r(x_c),v_next=v_bar_c)
        t += dt_c
    return x_f,x_c


@njit(nogil=True)
def max_np(A,axis=1):
    '''Returns array with maximum values along given axis. numba replace for np.max'''
    shape = np.shape(A)
    n = shape[1-axis]; m=shape[axis]
    out = np.zeros(n)
    for j in range(m):
        out = np.maximum(A[:,j],out)
    return out


def correlated_rv(x,v,dt_f,eps,v_tilde,N,M_t,B):
    '''Draw and calculate r.v. for next M_t fine steps and next coarse steps
    everything here is in standardized form
    '''
    dt_c = dt_f*M_t
    Z = np.random.normal(size=(N,M_t)); U = np.random.uniform(size=(N,M_t))
    C = (U>=eps**2/(eps**2+dt_f*r)) #Indicates if collisions happen


    z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1); u_c = np.max(U,axis=1)**M_t

@njit(nogil = True,parallel=True)
def test_var(dt_0,L,M_t,t,T,eps,N,Q,B):
    dt_list = dt_0/M_t**np.arange(0,L+1)
    runs = 10
    N_c = 10_000
    V = np.zeros((runs,len(dt_list)-1)); E = np.zeros((runs,len(dt_list)-1))
    V_d = np.zeros((runs,len(dt_list)-1)); E_bias = np.zeros((runs,len(dt_list)-1))
    for r in prange(runs):
        for i in range(1,L+1):
            x_f,x_c = correlated(dt_list[i],M_t,t,T,eps,N_c,Q,B)
            E_bias[r,i-1] = np.mean(x_f**2-x_c**2)
            V_d[r,i-1] = np.sum((x_f**2-x_c**2-E_bias[r,i-1])**2);
            E[r,i-1] = np.mean(x_f**2)
            V[r,i-1] = np.sum((x_f**2-E[r,i-1])**2)
            # V_d[i-1] = np.var(x_f**2-x_c**2); E_d[i-1] = abs(np.mean(x_f**2-x_c**2))
            # V[i-1] = np.var(x_f**2); E[i-1] = np.mean(x_f**2)
    E_bias = np.abs(E_bias)
    V_out,E_out = add_sum_of_squares_alongaxis(V,E,N_c)
    V_d_out,E_d_out = add_sum_of_squares_alongaxis(V_d,E_bias,N_c)
    return V_out/(N-1),E_out,V_d_out/(N-1),E_d_out

@njit(nogil=True,parallel=True)
def add_sum_of_squares_alongaxis(A,A_mean,N,axis=0):
    '''If axis is zero then the sum of squares of each coulmn is calculated
    A: matrix of sum of squares for each run
    A_mean: matrix of means for each run
    N: number of paths used in each run
    '''
    n = A.shape[axis]
    m = A.shape[0] if axis==1 else A.shape[1]
    ss = A[0,:]
    E = np.zeros(m)
    for j in prange(m):
        mu_old = A_mean[0,j]
        # print(mu_old)
        for i in range(1,n):
            Delta = A_mean[i,j] - mu_old
            M = (N*(N*(i)))/(N+N*(i))
            ss[j] = ss[j] + A[i,j] + Delta**2*M
            mu_old = mu_old + Delta*(N*i)/(N+N*i)
        E[j] = mu_old
    return ss,E


if __name__ == '__main__':
    model = 'Goldstein-Taylor'
    @njit(nogil=True)
    def B(x,model='Goldstein-Taylor'):
        '''
        Distribution of V_bar
        '''
        if model == 'Goldstein-Taylor':
            U = np.random.uniform(0,1,size=len(x))
            v_bar =  (U <= 0.5).astype(np.float64) - (U > 0.5).astype(np.float64)#np.random.normal(size=len(x))
        else:
            v_bar = np.ones(len(x))#np.random.normal(size=len(x))
            print('here')
        return v_bar,v_bar
    @njit(nogil=True)
    def Q(N):
        return (np.zeros(N),1,np.ones(N))
    # dt_f=0.2;M_t=5;t=0;T=10;N=10_000;eps=0.5
    # x_f,x_c = correlated(dt_f,M_t,t,T,eps,N,lambda N: (np.zeros(N),1,np.ones(N)),B,plot_var=True)
    if True:
        dt_0 = 2.5;M_t = 2;L=16;eps = 0.1
        dt_list = dt_0/M_t**np.arange(0,L+1)
        V,E,V_d,E_d = test_var(dt_0,L,M_t,0,5,eps,100_000,Q ,B)
        print(E_d)
        plt.figure(1)
        plt.subplot(122)
        plt.plot(dt_list[1:],V_d,label='Diff')
        plt.plot(dt_list[1:],V,label='Single')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Variance')
        plt.legend()
        plt.subplot(121)
        plt.plot(dt_list[1:],E_d,label='Diff')
        plt.plot(dt_list[1:],E,label='Single')
        plt.title('Mean')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

    if False:
        A = np.zeros((2,3))
        A_mean = np.array([[2,4,6],[0,0,0]])
        print(add_sum_of_squares_alongaxis(A,A_mean,N=1)[1])
