import numpy as np
from scipy.special import comb

input = np.array([0, 0, 0, 1, 0, 0, 0]) # initial distribution

N = 100 # number of individuals
a = np.array([0]*(2*N))
input = np.insert(a,49,1)
# transition matrix
P = np.zeros((2*N+1, 2*N+1))
for i in range(0,2*N+1):
    for j in range(0,2*N+1):
        P[i,j] = comb(2*N,j, exact=True, repetition=False)*(((i)/(2*N))**(j))*((1-(i)/(2*N))**(2*N-j))

n = 200 # number of steps to take
output = np.zeros((n+1,2*N+1)) # clear out any old values
t = np.arange(0,n)
output[0,:] = input  #generate first output value 
for i in range(1,n):
    output[i,:] = np.dot(output[i-1,:],P)
    if np.allclose(output[i,:],output[i-1,:]):
        print('Convergence after '+str(i),' iterations')
        break