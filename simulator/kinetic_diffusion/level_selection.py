import numpy as np

def select_levels(V,V_d):
    '''
    V: array of variances. Length L+1
    V_d: array of variances of bias. Length L
    '''
    l = 1
    test = V_d > V[1:]
    if np.sum(test)>1:
        l = np.argwhere(test).flatten()[-1]+2 #Last index where variance of bias is larger than V
    L = [l-1,l]
    V_min = V_d[max(l-2,0)]
    for j in range(l,len(V_d)):
        if V_d[j]<V_min/2:
            L += [j]
            V_min = V_d[j]
    return L
