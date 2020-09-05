from __future__ import print_function
__author__ = 'kmchugg'

import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import sample 
from matplotlib import pyplot
import statistics

C = 0.15 #maximum value of the target 
sample_data = []
p = [0.06,0.06,0.06,0.06,0.06,0.15,0.13,0.14,0.15,0.13]
I = np.ones(20)
q = 0.05*I
# print(p)
# print(q)

for i in range(0,len(p)):
    value_test = C*np.random.rand(1)*q[i]
    if(value_test <= p[i]):
        sample_data.append(q[i])
    else:
        pass

print(sample_data)
sample_mean = statistics.mean(sample_data)
sample_variance = statistics.variance(sample_data)
# fig = plt.figure()
# plt.hist(p,30,facecolor='b', alpha=1)
# plt.hist(sample_data,30,facecolor='r', alpha=1)
# plt.show()

bins = np.linspace(0, 0.2, 50)

plt.plot(p)
plt.axis([0 6 0 ])
pyplot.hist(sample_data, bins, alpha=0.5, label='sampled')
pyplot.legend(loc='upper right')
pyplot.show()