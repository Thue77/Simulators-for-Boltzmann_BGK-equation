import numpy as np
import matplotlib.pyplot as plt

A = [5.0]#,0.1,0.2,0.5,1.0,2.0,5.0,10.0]
B = [1.0,10.0,100.0,1000.0]
dt_list = 1/2**np.arange(0,22,1)
for b in B:
    plt.figure()
    for a in A:
        data = np.loadtxt(f'var_a_{a}_b_{b}_type_B1.txt')
        V = data[0]
        V_d = data[1]
        plt.plot(dt_list[1:],V_d[:-1],':', label = f'a={a}')
        plt.plot(dt_list,V,'--',color = plt.gca().lines[-1].get_color())
    plt.title(f'b = {b}, type: B1')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
plt.show()
