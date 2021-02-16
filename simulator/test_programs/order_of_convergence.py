import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

'''
test for dy/dt = cos(t),y(0)=0
'''

def f(x,y):
    return np.array([math.cos(x)])

def euler(x,y,f,dt):
    return (y + dt*f(x,y))

def heun(x,x_new,y,f,dt):
    y_temp = y + dt*f(x,y)
    return y + 0.5*dt*(f(x,y)+f(x_new,y_temp))


if __name__ == '__main__':
    DX = 1/(2**np.arange(0,15,1))
    y0 = np.array([0])
    end = 1
    error = np.zeros((2,len(DX)))
    for i,dx in enumerate(DX):
        x=0.0
        rk = RK45(f,x,y0,dx,max_step=1,first_step=dx)
        y_exact = math.sin(x)
        y_euler = euler(x,y0,f,dx)
        y_heun = heun(x,x+dx,y0,f,dx)
        rk.step()
        error[0,i] = np.abs(y_exact-y_euler)
        error[1,i] = np.abs(y_exact-rk.y)
    print(error)
    plt.plot(DX,error[0,:],label='euler')
    plt.plot(DX,error[1,:],label='heun')
    plt.plot(DX,DX,label='dx',linestyle='--')
    plt.plot(DX,DX**2,label='dx^2',linestyle='--')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
