from numba import njit
'''Script for initializing global variables in example script'''

@njit
def initialize():
    global a;global b;global type;global epsilon;
    a = 0; b = 1; type = 'default'; epsilon = 1
