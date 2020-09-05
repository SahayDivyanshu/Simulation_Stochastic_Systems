from __future__ import print_function
__author__ = 'kmchugg'

import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import sample 
from matplotlib import pyplot
import statistics
import collections

C = 0.15/0.05 #maximum value of the target 
accepted = 0
rejected = 0
sample_data = []
p = [0.06,0.06,0.06,0.06,0.06,0.15,0.13,0.14,0.15,0.13,0,0,0,0,0,0,0,0,0,0,]  #extra zeros added added to accoujt for the size of q
p_plot  = [0,0.06,0.06,0.06,0.06,0.06,0.15,0.13,0.14,0.15,0.13,0,0,0,0,0,0,0,0,0] #to plot the first value is taken as zero as the first value doesn't get printed.
I = np.ones(20)
q = 0.05*I
# print(p)
# print(q)

for i in range(10000):
    Y= random.randrange(1,21)
    value_test = C*np.random.rand(1)
    if(value_test <= p[Y-1]/0.05): # the algorithm
        sample_data.append(Y)
        accepted = accepted + 1
    else:
        rejected = rejected + 1
    
efficiency = accepted / (accepted + rejected)
print('---efficiency---')
print(efficiency)
print('--calculated---')
print(1/C)

print(sample_data)
value = []
freqList = (collections.Counter(sample_data)) #the disctionary that stores the frequency of occurrence of numbers
print(freqList)
denom = (len(sample_data))
for i in range(1, (len(freqList)+1)):
    value.append(freqList[i]/denom) #calculating the Probabilities
print('---value---')
print(value)

x = [1,2,3,4,5,6,7,8,9,10]
y = value
print('---x---')
print(x)


plt.figure(figsize=(10, 3))
plt.bar(x,y,align='center') # A bar chart
plt.plot(p_plot,"r-")
plt.xlabel('Numbers')
plt.ylabel('Probabilities')
plt.show()

sam_mean = statistics.mean(sample_data)
print('---sample mean---')
print(sam_mean)
sam_variance = statistics.variance(sample_data)
print('---sample variance---')
print(sam_variance)


theo_mean = 0
theo_var = 0
p_cal = [0.06,0.06,0.06,0.06,0.06,0.15,0.13,0.14,0.15,0.13]
for i in range(len(p_cal)):
    theo_mean = theo_mean + (i+1)*p_cal[i]
for i in range(len(p_cal)):
    theo_var = theo_var + (p_cal[i]*((i+1) - theo_mean)*((i+1) - theo_mean))
print('--theoretical mean---')
print(theo_mean)
print('--theoretical variance---')
print(theo_var)

fig = plt.figure()
plt.hist(p_plot,30,facecolor='b', alpha=1)
plt.hist(sample_data,30,facecolor='r', alpha=1)
plt.show()

