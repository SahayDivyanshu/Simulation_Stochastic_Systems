import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
X_samples = []
N= 1000
mu = 0
sigma  = 1
for i in range(N):
    z1 = np.random.normal(mu, sigma)
    z2 = np.random.normal(mu, sigma)
    z3 = np.random.normal(mu, sigma)
    z4 = np.random.normal(mu, sigma)
    X = z1*z1 + z2*z2 + z3*z3 + z4*z4
    X_samples.append(X)
X_samples.sort()
print(len(X_samples))
print(X_samples)
y = np.arange(0,1,1/N)
x = np.arange(0,15,10**(-4))
plt.plot(x, stats.chi2.cdf(x, df=4))
plt.step(X_samples,y,label='Empirical')
plt.show()
diff = 0
max_diff = 0
print('-- the maximum difference---')
for i in range(len(X_samples)):
    diff = abs(i*(1/N) - stats.chi2.cdf(X_samples[i], df=4))
    if(diff > max_diff):
        max_diff = diff
print(max_diff)
print('----25th percentile : Empirical ----')
print(np.percentile(X_samples,50)) 
print('----50th percentile : Empirical----')
print(np.percentile(X_samples,84))
print('----90th percentile : Empirical----')
print(np.percentile(X_samples,98.9))
print('----25th percentile : Theoretical ----')
print(np.percentile(x,25)) 
print('----50th percentile : Theoretical ----')
print(np.percentile(x,50))
print('----90th percentile : Theoretical ----')
print(np.percentile(x,90))

