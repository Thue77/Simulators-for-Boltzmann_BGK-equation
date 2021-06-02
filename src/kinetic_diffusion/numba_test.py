from numba import jitclass
import numpy as np

@jitclass

if __name__ == '__main__':
    print(second(np.array([1,2]),np.array([3,4])))
