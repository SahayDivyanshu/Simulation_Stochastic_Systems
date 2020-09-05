from __future__ import print_function
__author__ = 'kmchugg'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import sample 
import sys
# np.random.seed(10)
sum_N = 0
n=0
min=sys.maxsize
range1 = 100
range2 = 1000
range3 = 10000
list1=[]
for i in range(range3):
    n=0
    sum_N=0
    while 1:
        value = np.random.rand(1)
        sum_N = sum_N + value
        n = n + 1
        if (sum_N > 4):
            break
    list1.append(n)
    if (n < min):
        min = n 
print(min)         
print(list1)
avg = sum(list1)/len(list1)
print(avg)
fig = plt.figure()
plt.hist(list1,30,facecolor='b', alpha=1)
plt.show()




