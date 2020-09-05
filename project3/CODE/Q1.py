from __future__ import print_function
__author__ = 'kmchugg'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import sample 

n = 0
Lot = []
rejected = 0
prob_reject_list = []
for i in range(124):
    if (i < 6):
        Lot.append(0) # 0 DEPICTS A DEFECTIVE CHIP
    else:
        Lot.append(1) # 1 DEPICTS A DEFECTIVE CHIP

total_trial = 10000

for i in range(10):
    np.random.shuffle(Lot) #to simulate randomly placed microschips
    prob_reject = 0
    rejected = 0
    for i in range(total_trial):
        sample_check = sample(Lot,5)
        
        for i in range(0,len(sample_check)):
            if (sample_check[i] == 0):
                rejected = rejected + 1
                break
            else:
                pass
    prob_reject = rejected / total_trial
    prob_reject_list.append(prob_reject)        

print(prob_reject_list)

prob_reject = rejected / total_trial


fig = plt.figure()
plt.ylim(0, 0.3)
plt.plot(prob_reject_list,"r-")
plt.show()




 

