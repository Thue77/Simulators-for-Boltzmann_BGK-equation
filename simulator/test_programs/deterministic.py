import numpy as np
from scipy.stats import norm
from numpy.linalg import inv
'''Solve kinetic equation deterministically using central difference'''
eps = 1
#Step sizes
h_x = 0.1#input('step size for x: ')
h_v = 1#input('step size for v: ')

#Domain
start = (-1,-1) #coordinate of lower left corner in descrete domain
end = (1,1)

#Axis
x_axis = np.arange(start[0],end[0]+h_x,h_x)
v_axis = np.arange(start[1],end[1]+h_v,h_v)

#Number of increments
N_x = int((end[0]-start[0])/h_x)
N_v = int((end[1]-start[1])/h_v)

#Initial value - Try to initialise particles to by uniform in [-1,1] and let velocities be standard normal
'''At each x in [-5,5] generate standard normal numbers and count the th fraction
of numbers in each [v_i-h_v,v_i+h_v], and use that as an approximation at v_i'''
f = np.zeros((N_x+1)*(N_v+1)) #Vector f_hat
#initilise at time 0
for i in range(1,N_x):
    N = 10_000
    V = np.append(np.random.normal(loc=-1,size = int(N/2)),np.random.normal(loc=1,size = int(N/2)))
    v = start[1] + h_v
    for j in range(1,N_v):
        f[i+(N_v+1)*j] = (np.sum(np.logical_and(V<v+0.5*h_v,V>v-0.5*h_v))/N)/(N_x-1)
        v = v+h_v
'''Set up advection matrix. Needs to be a block diagonal matrix with the block diagonal defined as below'''
A = np.zeros(((N_x+1),(N_v+1)))
for i in range(1,N_x):
    A[i,i-1] = 1
    # A[i,i] = -2
    A[i,i+1] = 1
A = 1/h_x**2*A

'''Set up v'''
v = np.zeros((N_x+1)*(N_v+1))
for j in range(N_v+1):
    v[(N_x+1)*j:(N_x+1)*(j+1)] = start[1]+j*h_v

'''Convert f to matrix'''
def to_matrix(f,N_x,N_v):
    F = np.zeros((N_x+1,N_v+1))
    for i in range(N_x+1):
        F[i,:] = f[i*(N_x+1):(i+1)*(N_x+1)]
    return F

'''Block-diagonl multiplication'''
def block_multiply(A,f):
    out = np.zeros(len(f))
    for j in range(N_v+1):
        out[j*(N_x+1):(j+1)*(N_x+1)] = np.matmul(A,f[j*(N_x+1):(j+1)*(N_x+1)])
    return out

'''Calculate discrete densiyt values for M'''
def M(N=10_000):
    V = np.random.normal(size = int(N))
    out = np.zeros(N_v+1)
    for j in range(N_v+1):
        v_j = start[1]+j*h_v
        out[j] = np.sum(np.logical_and(V<v_j+0.5*h_v,V>v_j-0.5*h_v))/N
    return out


'''Calculate equilibrium state'''
def equilibrium(f,N_x,N_v):
    out = np.zeros(len(f))
    F = to_matrix(f,N_x,N_v)
    rho = np.sum(F,axis=0)
    M_v = M()
    for j in range(N_v+1):
        out[j*(N_x+1):(j+1)*(N_x+1)] = M_v[j]*rho
    return out



'''Use Backward for transport step and forward Euler for collision step
 to solve f_t + v/eps*f_x = 1/eps**2(M*rho-f)'''
I_t = (0,1)
dt = 0.01
steps = int((I_t[1]-I_t[0])/dt)
A_inv = np.linalg.inv(np.identity((N_x+1)*(N_v+1))+dt*block_multiply(A,v)/eps)
for i in range(steps):
    f = np.matmul(A_inv,f) #Transport step
    f = f+ dt/eps**2*(equilibrium(f,N_x,N_v)-f) #Collision step
print(f)
