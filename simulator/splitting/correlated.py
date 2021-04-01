import numpy as np
from .one_step import phi_APS
import matplotlib.pyplot as plt
from numba import njit,jit_module,prange

@njit(nogil=True)
def correlated(dt_f,M_t,t,T,eps,N,Q,B,r=1,plot=False,plot_var=False):
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
            x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r)
            v_bar_c[C[:,m]] = v_f[C[:,m]]
        z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c,_ = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c)
        t += dt_c
    return x_f,x_c
def correlated_test(dt_f,M_t,t,T,eps,N,Q,B,r=1,plot=False,plot_var=False):
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
        X_f = [x_f]
        X_c = [x_c]
        C1 = [(0.0,np.array([0.0]))]
        C2 = [(0.0,np.array([0.0]))]
    while t<T:
        Z = np.random.normal(0,1,size=(N,M_t)); U = np.random.uniform(0,1,size=(N,M_t))
        C = (U>=eps**2/(eps**2+dt_f*r)) #Indicates if collisions happen
        # print(f'probability = {1-eps**2/(eps**2+dt_f*r)}')
        for m in range(M_t):
            x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r)
            v_bar_c[C[:,m]] = v_f[C[:,m]]#v_bar_f[C[:,m]]
            # print(f'v_last: {v_last}')
            if plot:
                X_f += [x_f]
                if C[:,m]: C1 += [(t+(m+1)*dt_f,x_f)]
        z_c = 1/np.sqrt(M_t)*np.sum(Z,axis=1)
        u_c = max_np(U,axis=1)**M_t
        x_c,v_c,_ = phi_APS(x_c,v_c,dt_c,eps,z_c,u_c,B,r=r,v_next=v_bar_c)
        t += dt_c
        if plot:
            X_c += [x_c]
            if u_c >=eps**2/(eps**2+dt_c*r): C2 += [(t,x_c)]
        if plot_var:
            var_d += [np.var(x_f-x_c)]
            var_f += [np.var(x_f)]
            var_c += [np.var(x_c)]
    if plot:
        plt.plot(np.arange(0,T+dt_f,dt_f),X_f,'.-')
        plt.plot(np.arange(0,T+dt_c,dt_c),X_c,'.-')
        plt.plot([t for t,_ in C1],[x[0] for _,x in C1],'x',color='blue')
        plt.plot([t for t,_ in C2],[x[0] for _,x in C2],'x',color='red')
        print(f'C1: {C1} \n C2: {C2}')
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
