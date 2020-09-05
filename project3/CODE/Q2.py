
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import poisson


p = 120/5000
arrival = []
arrival_list = []


# for i in range(100):
for i in range(1000):
    arrival_sum = 0
    arrival = []
    random_prob =  np.random.rand(5000)
    for i in range(5000):
        if( p >= random_prob[i]):
            arrival.append(1)

    # print(arrival)
    arrival_sum = sum(arrival)
    arrival_list.append(arrival_sum)
print(arrival_list)

fig = plt.figure()
plt.figure(figsize=(10, 3))
x= np.arange(50,200)
scale = poisson.pmf(x,120)
plt.plot(x, scale*2230, "r-")
plt.hist(arrival_list,50,facecolor='b', alpha=1)
plt.show()

# arrival_list.append(arrival_sum)

# print(arrival_list)


