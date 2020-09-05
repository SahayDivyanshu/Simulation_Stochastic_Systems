import random
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
a = -2
b = 2
N  = 1000
result = 0
law = 0
final_result = []
def func(X):
        return math.exp(X + X*X) #THE ACTUAL FUNCTION 
for i in range(N):
    total_samples = np.zeros(N)
    law = 0
    for i in range (len(total_samples)):
        total_samples[i] = random.uniform(a,b) #GENERATING RANDOM VALUES BETWEEN -2 and 2
    for i in range(N):
        law = law + func(total_samples[i]) 
    result = (b-a)*(law/N) #USING THE LAW OF LARGE NUMBERS TO APPROXIMATE THE EXPECTATION
    final_result.append(result)
fig = plt.figure()
plt.hist(final_result)
plt.show()