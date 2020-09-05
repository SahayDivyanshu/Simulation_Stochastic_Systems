import random
import math
import numpy as np
import scipy.stats as stats
import statistics 
import matplotlib.pyplot as plt

a = -1
b = 1
N  = 1000
result = 0
law = 0
final_result = []

# def func(X,Y):
#     k = np.exp(5*abs(X-5) + 5*abs(Y-5))
#     return k

def func(X,Y):
    k = math.cos(math.pi + 5*X + 5*Y)
    return k

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
plt.title('1b')
plt.xlabel('Result Obtained')
plt.ylabel('Frequecy')
plt.hist(final_result)
print('Variance')
print(statistics.variance(final_result))
plt.show()