import numpy as np


'''Script for setting up the discrete domain in which the solution is approximated.
    The approximation is done via a histogram approach, i.e. the value of rho(x_i,t=T)
    is the number of particles in [x-0.5*dx,x+0.5*dx)'''

class Omega():
    """Omega represents the discretized spacial domain. Implementation is for
    a domain (x,v) in RxV."""

    def __init__(self, x_lim, v_lim, N_x,N_v,mp=1):
        '''
        x_lim: tuple of boundaries for x-axis
        v_lim: tuple of boundaries for v-axis
        N_x/N_v: number of interior mesh points on the x-axis/v-axis. Number of intervals
                is 1 more!
        m_p: mass of particles
        '''
        self.x_0,self.x_L = x_lim
        self.v_0,self.v_L = v_lim
        self.N_x = N_x; self.N_v = N_v
        self.dx = (self.x_L-self.x_0)/(self.N_x+1)
        self.dv = (self.v_L-self.v_0)/(self.N_v+1)
        self.x_axis = np.arange(self.x_0+self.dx,self.x_L,self.dx)
        self.mp=mp


    '''Estimating density in Omega'''
    def density_estimation(self,x):
        '''
        x: array of particle positions
        '''
        N = len(x) # Number of particles
        rho = np.zeros(self.N_x)
        for i in range(self.N_x):
            rho[i] = self.mp/N*np.sum(np.logical_and(self.x_axis[i]-0.5*self.dx<=x, x<= self.x_axis[i]+0.5*self.dx))
        return rho

    '''Discretize function in Omega. Only works for interior points'''
    def discrete_f(self,f):
        return None
