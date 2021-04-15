import numpy as np
from .correlated import correlated


def select_levels(t0,T,M_t,strategy=1,cold_start=True,N=100):
    '''
    strategy: indicates if strategy 1 or 2 is used
    cold_start: indicates if the variance and estimates are calculated for an
    initial number of paths
    '''
    levels = []
    if strategy==1:
        levels += [eps**2]
        levels += [eps**2/M_t]
        N_out=np.ones(2)*N
        N_diff=np.ones(2)*N
        V_out = np.zeros(2);C_out = np.zeros(2);
    elif strategy==2:
        levels += [T-t0]
        levels += [eps**2]
        levels += [eps**2/M_t]
        N_out=np.ones(3)*N
        N_diff=np.ones(3)*N
        V_out = np.zeros(3);C_out = np.zeros(3);
    return levels,N_out,Q_out,V_out,C_out


def ml(e2,Q,t0,T,M_t,eps,M,r,F):
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
