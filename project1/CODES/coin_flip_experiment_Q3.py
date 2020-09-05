from __future__ import print_function
__author__ = 'kmchugg'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#N_students = 200 # This can be changed to change the numer of times the experiment is performed
N_flips = 125
p = 6.0/125.0
head_count = 0
longest_run = 0
head_count_list = []
np.random.seed(10)  
toss = np.random.rand(N_flips)
for i in range(toss.shape[0]):
    if (toss[i] < p):
        toss[i] = 0
        if(head_count != 0):
            head_count_list.append(head_count) # storing the head count in the list
        head_count = 0 
    else:
        toss[i] = 1
        head_count = head_count + 1
if(head_count != 0):
    head_count_list.append(head_count)        # storing the head count in the list
        
print(head_count_list)        
plt.hist(head_count_list,12) # plotting the histogram
plt.show()

        
#print("longest run of heads: ", longest_run)    
print(toss)
#x = range(1, N_flips+1)
#plt.hist(toss,2)
#plt.show()








# heads = np.arange(0,N_flips+1)
# binomial_pmf = stats.binom.pmf(heads, N_flips, p)

# flip_results = np.array([
#                     # <<---- add numbers here to "force" a particular class
#     ])

# # generate samples if no class data
# if (flip_results.shape[0] == 0):
#     flip_results=np.random.binomial(N_flips, p, N_students)

# print("flip_results")
# print(flip_results)

# flip_counts= np.zeros(N_flips+1) #[0 for i in range(N_flips+1)]
# for k in range(N_flips+1):
#     flip_counts[k]=(sum(flip_results==k))

# print("flip_counts")
# print(flip_counts)
# print(sum(flip_counts))

# print("number of students =",N_students)
# print("number of flips =",N_flips)
# print("probability of heads =",p)

# print("\n\nk n(k) p_hat(k) MOE(k)\n")

# p_hat = flip_counts / float(N_students)
# MOE = np.zeros( (2,N_flips+1) )
# MOE[0] = 1.96 * np.sqrt( p_hat * (1.0 - p_hat) / N_students  )

# for k in range(len(p_hat)):
#     if flip_counts[k] == 0 :
#         MOE[1][k] = 3.0 / N_students
#     else:
#         MOE[1][k] = MOE[0][k]
#     print(k,flip_counts[k],"{:0.4g}".format(p_hat[k]),"{:0.4g}".format(MOE[0][k]),sep=" ")

# print("\n\n")
# for k in range(len(p_hat)):
#     print(p_hat[k])

# fig = plt.figure()
# plt.stem(heads, binomial_pmf, 'r', markerfmt='ro', label='model' )
# plt.errorbar(heads, p_hat,  yerr=MOE, fmt='o', label='measured')
# plt.legend(loc=2)
# plt.show()


# fig = plt.figure()
# plt.hist(flip_results, 40, density=True, facecolor='b', alpha=1)
# plt.show()



