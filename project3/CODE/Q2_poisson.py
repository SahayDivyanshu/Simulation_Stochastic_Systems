import math
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import poisson

sampled = []
lam_da = 120
for j in range(1000):
    i = 0
    p = math.exp(-lam_da)
    F = p
    random_prob = np.random.rand(1)
    while(random_prob >= F):
        i = i + 1
        p = (lam_da * p)/(i)
        F = F + p
        # print("phanse")
    sampled.append(i)

print(sampled)
fig = plt.figure()
plt.figure(figsize=(10, 3))
x= np.arange(50,200)
scale = poisson.pmf(x,120)
plt.plot(x, scale*2200, "r-")
plt.hist(sampled,50,facecolor='b', alpha=1)
plt.show()