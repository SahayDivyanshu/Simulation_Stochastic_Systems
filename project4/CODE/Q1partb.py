import random
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

a = 1
b = 0
N  = 1000
result = 0
law = 0
final_result = []

def func(X):
    k1 = -((1-X)/X)*((1-X)/X)
    k2 = math.exp(k1)*(X-1)
    k3 = k2/(X*X*X)
    return k3

for i in range(N):
    total_samples = np.zeros(N)
    law = 0
    for i in range (len(total_samples)):
        total_samples[i] = random.uniform(a,b)
 
    for i in range(N):
        law = law + func(total_samples[i])

    result = (b-a)*(law/N)
    final_result.append(math.sqrt(2*math.pi*result))

fig = plt.figure()
plt.hist(final_result)
plt.show()