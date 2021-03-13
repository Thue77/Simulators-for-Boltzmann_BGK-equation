import numpy as np
import matplotlib.pyplot as plt

A = [0.0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0]
B = [1.0,10.0,100.0,1000.0]
dt_list = 1/2**np.arange(0,22,1)
# fig,axes = plt.subplot(nrow=2,ncol=2)
plt.figure(1)

for i,b in enumerate(B):
    index = 221 + i
    plt.subplot(index)
    for a in A:
        data = np.loadtxt(f'var_a_{a}_b_{b}_type_B1.txt')
        V = data[0]
        V_d = data[1]
        plt.plot(dt_list[1:],V_d[:-1],':', label = f'a={a}')
        plt.plot(dt_list,V,'--',color = plt.gca().lines[-1].get_color())
    plt.title(f'b = {b}')
    plt.xscale('log')
    plt.yscale('log')
# plt.legend(loc=9)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0)
plt.legend(bbox_to_anchor=(-0.6,2.3,1,0.2), loc="upper center",
                mode="expand", borderaxespad=0, ncol=3)
plt.show()
