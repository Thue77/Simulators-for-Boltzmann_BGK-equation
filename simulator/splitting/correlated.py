import numpy as np
from one_step import phi_APS,v_char
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
    x_f,_,v_bar_f = Q(N)
    v_f = v_char(dt_f,eps)*v_bar_f; v_c = v_char(dt_c,eps)*v_bar_f
    x_c = x_f.copy()
    v_bar_c = v_bar_f.copy()
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
        C = (U>=eps**2/(eps**2+dt_f*r)) #Indicates if collisions hppen
        for m in range(M_t):
            x_f,v_f,v_bar_f = phi_APS(x_f,v_f,dt_f,eps,Z[:,m],U[:,m],B,r=r)
            v_bar_c[C[:,m]] = v_bar_f[C[:,m]]
            # print(f'v_last: {v_last}')
            if plot:
                X_f += [x_f]
                if C: C1 += [(t+(m+1)*dt_f,x_f)]
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
    # if plot:
    #     plt.plot(np.arange(0,T+dt_f,dt_f),X_f,'.-')
    #     plt.plot(np.arange(0,T+dt_c,dt_c),X_c,'.-')
    #     plt.plot([t for t,_ in C1],[x[0] for _,x in C1],'*',color='blue')
    #     plt.plot([t for t,_ in C2],[x[0] for _,x in C2],'*',color='red')
    #     plt.grid(color='black', lw=1.0)
    #     plt.show()
    # if plot_var:
    #     plt.plot(np.arange(0,11),var_d,'.-',label='Diff')
    #     plt.plot(np.arange(0,11),var_f,'.-',label='Fine')
    #     plt.plot(np.arange(0,11),var_c,'.-',label='Coarse')
    #     plt.legend()
    #     plt.show()
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
    V_d = np.zeros(L); E_d = np.zeros(L);
    V = np.zeros(L); E = np.ones(L);
    for j in prange(10):
        N_c = 10_000
        for i in range(1,L+1):
            x_f,x_c = correlated(dt_list[i],M_t,t,T,eps,N_c,Q,B)
            V_d[i-1] = np.var(x_f**2-x_c**2); E_d[i-1] = np.mean(x_f**2-x_c**2)
            V[i-1] = np.var(x_f**2); E[i-1] = np.mean(x_f**2)
    return V,E,V_d,E_d

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
        return v_bar
    @njit(nogil=True)
    def Q(N):
        return (np.zeros(N),1,np.ones(N))
    # dt_f=0.2;M_t=5;t=0;T=10;N=10_000;eps=0.5
    # x_f,x_c = correlated(dt_f,M_t,t,T,eps,N,lambda N: (np.zeros(N),1,np.ones(N)),B,plot_var=True)
    if True:
        dt_0 = 2.5;M_t = 2;L=16
        dt_list = dt_0/M_t**np.arange(0,L+1)
        V,E,V_d,E_d = test_var(dt_0,L,M_t,0,5,0.1,100_000,Q ,B)
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
