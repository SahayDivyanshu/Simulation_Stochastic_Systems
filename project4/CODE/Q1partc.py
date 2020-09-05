import random
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

a = 0
b = 1
N  = 1000
result = 0
law = 0
final_result = []

def func(X,Y):
    k = (X+Y)*(X+Y)
    p = - k
    return math.exp(p)

for i in range(N):
    total_samplesX = np.zeros(N)
    total_samplesY = np.zeros(N)
    law = 0
    for i in range (len(total_samplesX)):
        total_samplesX[i] = random.uniform(a,b)
        total_samplesY[i] = random.uniform(a,b)
 
    for i in range(N):
        law = law + func(total_samplesX[i],total_samplesY[i])

    result = (b-a)*(b-a)*(law/N)
    final_result.append(result)

# print(final_result)
fig = plt.figure()
plt.hist(final_result)
plt.show()