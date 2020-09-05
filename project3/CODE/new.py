from __future__ import print_function
__author__ = 'kmchugg'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import sample 
import statistics


sum_60 = 0
for i in range(61):
    if(i==0):
        pass
    else:
        sum_60 = sum_60 + (1/i)
p = 1 / sum_60        # calculating the value of p
print(sum_60)
print(p)

sum_prob_list = []
sum_prob = 0
X_k = []
N_60 = 0
N_60_list = []
Expect = 0

for i in range(60):
    sum_prob = sum_prob + p/(i+1)
    sum_prob_list.append(sum_prob)
print(sum_prob_list) # calculating the vector with specified probabilities

for i in range(1000):#peforming the experiment 1000 times
    X_k = []
    N_60 = 0
    for i in range(1000): # creating a sequence of 1000 numbers 
        u = np.random.rand(1)    
        for i in range(len(sum_prob_list)):
            if (u < sum_prob_list[i]): #performing the discrete inverde transform method to generate the samples.
                X_k.append(i+1)
                break
    for i in range(len(X_k)):
        if (X_k[i] == 60): #checking for the occurrane of 60
            N_60 = i
            break
    N_60_list.append(N_60)
print(len(X_k))
print(N_60_list) #the list with the numbers of 60 in each sequence generated
MEAN = statistics.mean(N_60_list)
VARIANCE = statistics.variance(N_60_list)
print('---MEAN---')
print(MEAN)
print('---VARIANCE---')
print(VARIANCE)
fig = plt.figure()
plt.hist(X_k,50,facecolor='b', alpha=1)
plt.show()

        

