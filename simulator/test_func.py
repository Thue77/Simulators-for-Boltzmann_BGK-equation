import numpy as np
from numba import prange
from scipy.integrate import quad

def integral_of_R(R,t_c,t_f,x,v):
    '''
    This function calculates integrals of R from t_f to t_c if t_f<t_c.
    This is to calculate exponential numbers for the coarse path.
    t_f,t_c: numpy arrays of start and end times
    '''
    index = np.argwhere(t_c>t_f).flatten()
    I = np.zeros(len(t_c),dtype=np.float64)
    sign = np.sign(x+v*(t_c-t_f)-x)
    I[index] = np.zeros(len(index))
    for j in prange(len(index)):
        i = index[j]
        # I[i] = R(x[i])*(t_c[i]-t_f[i])
        start = min(x[i],x[i]+v[i]*(t_c[i]-t_f[i]))
        end = max(x[i],x[i]+v[i]*(t_c[i]-t_f[i]))
        dx = min(1e-6,end-start)
        pos = start
        while pos < end:
            I[i] = I[i] + R(pos+dx/2)*dx
            pos += dx
        I[i]=I[i]/v[i]*sign[i]
    return I

def I_R(R,t_c,t_f,x,v):
    '''This function calculates integrals of R from a to b if a<b
    a,b: numpy arrays of start and end times
    '''
    index = np.argwhere(t_c>t_f).flatten()
    I = np.zeros(len(t_c))
    for i in index:
        I[i] = 1/v[i]*quad(R,x[i],x[i]+v[i]*(t_c[i]-t_f[i]))[0]
    return I

if __name__ == '__main__':
    x = np.array([3,-3]); v = np.array([2,-2]); t_f=np.array([0.3,0.05]);t_c=np.array([0.35,0.08])
    R = lambda x: 6*x+2
    print(f'Scipy integral: {I_R(R,t_c,t_f,x,v)}, my integral: {integral_of_R(R,t_c,t_f,x,v)}')
