import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
'''Script for plotting values in any given resultfile from the folder Logfiles'''

parser = argparse.ArgumentParser()
parser.add_argument('name', help='name of the relevant resultfile', type=str)
parser.add_argument('folder', help='name of the folder with relevant resultfile', type=str)
args = parser.parse_args()
if args.folder:
    print(os.path.realpath(args.folder))
path = os.getcwd()
path += '\'
path += args.folder
os.chdir(path)

(dt_list,v,b,var1,var2,cost1,cost2,kur1,cons) = np.loadtxt(args.name)
L = dt_list.size
plt.plot(dt_list[1:],var2[1:],':',label='var(F(x^f)-F(X^c))')
plt.plot(dt_list,var1,'--',color = plt.gca().lines[-1].get_color(),label='var(F(X))')
plt.plot(dt_list[1:],np.abs(b[1:]),':',label='mean(|F(x^f)-F(X^c)|)')
plt.plot(dt_list,v,'--',color = plt.gca().lines[-1].get_color(),label='mean(F(X))')
plt.title(f'Plot of variance and bias')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.figure()
plt.plot(range(1,L),kur1[1:],':',label='kurtosis')
plt.title(f'Plot of kurtosis')
plt.figure()
plt.plot(range(1,L),cons[1:],':',label='kurtosis')
plt.title(f'Plot check of consistency')
plt.show()
