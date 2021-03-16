import numpy as np
from numba import prange
from scipy.integrate import quad

type = 'B1'
epsilon=2
a=1;b=1

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

def R(x):
    if type == 'default':
        return 1/(epsilon**2)
    elif type == 'B1':
        return -b*(a*(x-1)-1)*(x<=1) + b*(a*(x-1)+1)*np.logical_not(x<=1)

def I_R(R,t_c,t_f,x,v):
    '''This function calculates integrals of R from a to b if a<b
    a,b: numpy arrays of start and end times
    '''
    index = np.argwhere(t_c>t_f).flatten()
    I = np.zeros(len(t_c))
    for i in index:
        I[i] = 1/v[i]*quad(R,x[i],x[i]+v[i]*(t_c[i]-t_f[i]))[0]
    return I

#Anti derivative of R
def R_anti(x):
    if type == 'default':
        return x/(epsilon**2)
    elif type == 'B1':
        return (-b*a/2*x**2 + (a+1)*b*x)*(x<=1) + (b*a/2*x**2+(1-a)*b*x)*(x>1)
def I_anti(R_anti,t_c,t_f,x,v):
    start = np.minimum(x,x+v*(t_c-t_f))
    end = np.maximum(x,x+v*(t_c-t_f))
    # #Check if they are in the same domain and that coarse path is ahead of fine path
    index = np.argwhere(np.logical_and((start<=1)==(end<=1),t_c>t_f)).flatten()
    I = np.zeros(len(x))
    I[index] = (R_anti(end[index]) - R_anti(start[index]))/np.abs(v[index])
    #Find particles where they move into different domain
    index = np.argwhere(np.logical_and((start<=1)!=(end<=1),t_c>t_f)).flatten()
    I[index] = ((R_anti(end[index])-R_anti(1+1e-15))+(R_anti(1)- R_anti(start[index])))/np.abs(v[index])
    print(start); print(end)
    # print(f'start: {start}, end: {end}')
    # print(f'R_anti(start)={R_anti(start)},R_anti(end)={R_anti(end)}')
    # print(f'R_ant(1)= {R_anti(1)}')
    return I#(R_anti(end) - R_anti(start))/np.abs(v)

def compute_mean_alongaxis(A,axis=0):
    '''If axis is zero then the mean of each coulmn is calculated'''
    n = A.shape[axis==1]
    m = A.shape[axis!=1]
    mu = np.zeros(m)
    for j in prange(m):
        for i in prange(n):
            mu[j] = mu[j] + 1/(i+1)*(A[i,j]-mu[j])
    return mu

if __name__ == '__main__':
    x = np.array([1.217901707685121,-3]); v = np.array([-1.4764421219954438,-2]); t_f=np.array([0.3,0.05]);t_c=np.array([0.3+0.5242207430263304,0.08])
    # R = lambda x: 6*x+2
    print(f'Scipy integral: {I_R(R,t_c,t_f,x,v)}, my integral: {integral_of_R(R,t_c,t_f,x,v)}, with antiderivative: {I_anti(R_anti,t_c,t_f,x,v)}')
    # A = np.array([[1,2,3,4],[2,3,2,1],[5,3,7,1]])
    # print(f'numpy mean: {np.mean(A,axis=0)}, my mean = {compute_mean_alongaxis(A)}')
