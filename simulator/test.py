import numpy as np
import sys
from numba import njit

# @njit(nogil=True)
def cor_rv(M_t,C,v_bar_all):
    '''Determine influence of each v*'''
    '''The number of 1's in C_a indicate the number of steps that the first velocity
    influences and the number 2's indicate the number of steps that the second
    velocity influences'''
    C_a = np.zeros_like(v_bar_all)
    # print(C)
    n = C.shape[0]
    for j in range(n):
        C_a[j,:] = np.cumsum(C[j,:])
    # C_a = np.cumsum(C,axis=1)
    # print(f'C_a= {C_a}')

    #number of steps affected by collisions
    steps = np.count_nonzero(C_a>0,axis=1)
    #number of steps that each collision affects. count[0,1]: number of steps affected by second collision for path 0
    temp = np.zeros_like(v_bar_all).astype(np.int64)
    # i_c = np.zeros(M_t,dtype=np.int64)
    for i in range(M_t):
        temp[:,i] = np.count_nonzero(C_a==i+1,axis=1)
        # i_c = np.sum(count[:,0:i])
    # print(f'temp: {temp}')
    #fit index of number of steps affected by collision with the index of the collision
    start = np.minimum(M_t-np.sum(temp,axis=1),M_t).astype(np.int64)
    not_done = start<M_t
    count = np.zeros_like(v_bar_all).astype(np.int64)
    put_np(count,temp,M_t)
    # for i in range(M_t):
    #     # print(f'start: {start}')
    #     count[not_done,start] = temp[not_done,i]
    #     print(f'count: {count[not_done,start]}')
    #     print(f'temp: {temp[not_done,i]}')
    #     # sys.exit()
    #     old = start.copy()
    #     start = np.minimum(start+temp[:,i],M_t-1).astype(np.int64)
    #     not_done = old!=start


    print(f'count={count}, steps = {steps}')
    theta = np.ones_like(v_bar_all)
    '''Cannot divide with steps for paths where no collisions occurs'''
    index = np.where(steps>0)[0]
    theta[index,:] = (count[index,:].T/steps[index]).T

    # print(theta)
    # print(f'steps: {steps},\n C_a: {C_a},\n theta: {theta}\n v_bar_all: {v_bar_all}')
    # print(f'output: {np.sum(np.sqrt(theta)*v_bar_all,axis=1)}')

    return np.sum(np.sqrt(theta)*v_bar_all,axis=1)

def put_np(count,temp,M_t):
    print(f'temp: {temp}')
    for i in range(count.shape[0]):
        start = M_t-np.sum(temp[i,:])
        for j in range(M_t):
            if start>=M_t:
                break
            count[i,start] = temp[i,j]
            start = start + temp[i,j]
    # return count


if __name__ == '__main__':

    M_t = 5;
    C = np.array([[False, False,  True, False,  True],[False, False,  True, False,  True]])
    v_bar_all = np.array([[ 0.,          0.,         -0.4226053,   0.,          0.93366715],[ 0.,          0.,         -0.4226053,   0.,          0.93366715]])
    print(f'result: {cor_rv(M_t,C,v_bar_all)}')
