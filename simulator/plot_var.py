import numpy as np
import matplotlib.pyplot as plt
import math
import os

A = [0.0,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0]
B = [1.0,10.0,100.0,1000.0]
dt_list = 1/2**np.arange(0,22,1)

'''x-values for slopes'''


# fig,axes = plt.subplot(nrow=2,ncol=2)
plt.figure(1)
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
for i,b in enumerate(B):
    index = 221 + i
    plt.subplot(index)
    for a in A:
        rel_path = f'var_results/var_a_{a}_b_{b}_type_B1.txt'
        abs_file_path = os.path.join(script_dir, rel_path)
        data = np.loadtxt(abs_file_path)
        V = data[0]
        V_d = data[1]
        # PLot slop marker
        if a == 0:
            index = 13 if b>1 else 4
            x2 = dt_list[index]; x1 = dt_list[16]
            y2 = V_d[index-1]; y1 = V_d[15]
            slope = (math.log10(y2)-math.log10(y1))/(math.log10(x2)-math.log10(x1))
            print(round(slope))
            plt.plot(x1+np.array([1e-4,3e-4,6e-4,6e-4]),np.tile(y1,4),color='black')
            l = 6e-4-1e-4
            t = math.log10(y1) + 3*(math.log10(x1+6e-4)-math.log10(x1+1e-4)) # top of triangle
            y = np.linspace(y1,10**t,5)
            plt.plot(np.tile(x1+6e-4,5),y,color='black') #plot vertical line
            plt.text(x1+3e-4,y[0]+0.00625*(y[-1]-y[0]),f'{round(slope)}')
            plt.plot(x1+np.array([1e-4,3e-4,6e-4,6e-4]),(x1+np.array([1e-4,3e-4,6e-4,6e-4]))**3*(y1/(x1+1e-4)**3),color='black') #plot diagonal line
        plt.plot(dt_list[1:],V_d[:-1],':', label = f'a={a}')
        plt.plot(dt_list,V,'--',color = plt.gca().lines[-1].get_color())

    plt.title(f'b = {b}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('\u0394 t')
    plt.ylabel('variance')
# plt.legend(loc=9)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0)
plt.legend(bbox_to_anchor=(-0.6,2.3,1,0.2), loc="upper center",
                mode="expand", borderaxespad=0, ncol=3)
plt.show()
