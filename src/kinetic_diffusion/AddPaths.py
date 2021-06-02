import numpy as np
import math
from numba import njit,jit_module,prange,objmode

def delta(x_hat_1,x_hat_2,M1,M2):
    if M1 >= M2:
        return x_hat_2-x_hat_1
    else:
        return x_hat_1-x_hat_2
# Function is for the sum of squares
# Sfunc= lambda M1,M2,S_1,S_2,delta:

def Sfunc(M1,M2,S_1,S_2,delta):
    return S_1+S_2+delta**2*M1*M2/(M1+M2)

def x_hat(M1,M2,x_hat_1,x_hat_2,Delta):
    if M1>=M2:
        return x_hat_1+Delta*M2/(M1+M2)
    else:
        return x_hat_2+Delta*M1/(M1+M2)

jit_module(nopython=True,nogil=True)
