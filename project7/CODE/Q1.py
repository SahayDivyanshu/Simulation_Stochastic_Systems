import numpy as np
import matplotlib.pyplot as plt
N = 1000
mu1, sigma1 = -1, 1 # mean and standard deviation of the two models
mu2, sigma2 = 1, 1
p = 0.4
s = np.zeros(N)


for i in range(N):
    if np.random.rand() < p:
        s[i] = np.random.normal(mu1, sigma1)
    else:
        s[i] = np.random.normal(mu2, sigma2)

count, bins, ignored = plt.hist(s, 20, density=True)
plt.plot(bins, p*(1/(sigma1 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu1)**2 / (2 * sigma1**2) )) + (1-p)*(1/(sigma2 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu2)**2 / (2 * sigma2**2)) ),
         linewidth=2, color='r')
plt.show()

