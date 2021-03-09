import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''Program to test properties of poisson distribution'''

'''Test the distribution of tau_2 given K=10 and K=25. Event time k is distributed
    according the the k'th order statistic of uniform r.v., see https://www.uni-muenster.de/Stochastik/lehre/WS1314/BachelorWT/Daten/StPro_Ross1.pdf'''

#Generate event times for K events
K=10
# lmbda = 10
dt = 0.5
numbers=10_000
t = np.arange(0,dt+dt/numbers,dt/numbers)
tau2 = np.zeros(numbers)

for i in range(numbers):
    U = np.random.uniform(0, 1, K)
    T = dt*np.sort(U)
    tau2[i] = T[5] -T[4]


def tau_K(K,dt,t):
    return K*(dt-t)**(K-1)/dt**K

print(f'length of tau: {len(tau2)}, length of t: {len(t)}')

# df = pd.DataFrame({'Samples':np.hstack((tau2,tau_K(K,dt,t[1:]))),'Category':['simulation' if i < numbers else 'theoretical' for i in range(2*numbers) ]})
df = pd.DataFrame({'Samples':tau2,})

sns.displot(data=df,x='Samples',kind='kde')
plt.plot(t,tau_K(K,dt,t),color='yellow')

# penguins = sns.load_dataset("penguins")
# sns.displot(data=penguins, x="flipper_length_mm")
#
# print(df)
# print(penguins)
plt.show()
