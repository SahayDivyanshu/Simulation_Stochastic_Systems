from __future__ import print_function
__author__ = 'kmchugg'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

N_students=29 # <<---- change this to adjust class size
N_flips=10
p= 1.0 / 2.0
np.random.seed(123)  # <<---- change this to get a "new class"

heads = np.arange(0,N_flips+1)
binomial_pmf = stats.binom.pmf(heads, N_flips, p)

flip_results = np.array([
                    # <<---- add numbers here to "force" a particular class
    ])

# generate samples if no class data
if (flip_results.shape[0] == 0):
    flip_results=np.random.binomial(N_flips, p, N_students)

flip_counts= np.zeros(N_flips+1) #[0 for i in range(N_flips+1)]
for k in range(N_flips+1):
    flip_counts[k]=(sum(flip_results==k))

print("number of students =",N_students)
print("number of flips =",N_flips)
print("probability of heads =",p)

print("\n\nk n(k) p_hat(k) MOE(k)\n")

p_hat = flip_counts / float(N_students)
MOE = np.zeros( (2,N_flips+1) )
MOE[0] = 1.96 * np.sqrt( p_hat * (1.0 - p_hat) / N_students  )

for k in range(len(p_hat)):
    if flip_counts[k] == 0 :
        MOE[1][k] = 3.0 / N_students
    else:
        MOE[1][k] = MOE[0][k]
    print(k,flip_counts[k],"{:0.4g}".format(p_hat[k]),"{:0.4g}".format(MOE[0][k]),sep=" ")

print("\n\n")
for k in range(len(p_hat)):
    print(p_hat[k])

fig = plt.figure()
plt.stem(heads, binomial_pmf, 'r', markerfmt='ro', label='model' )
plt.errorbar(heads, p_hat,  yerr=MOE, fmt='o', label='measured')
plt.legend(loc=2)
plt.show()



