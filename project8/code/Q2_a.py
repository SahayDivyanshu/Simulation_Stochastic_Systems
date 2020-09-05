import numpy as np 
import random
import matplotlib.pyplot as plt
X1_list = []
X2_list = []
X3_list = []
X_sum_list = []
# part A
# lets choose starting point for the simulation as follows:
X1 = 5
X2 = 7
for i in range(1000):
    # as we have restricted our sampling now X3 can only be slected strcitly less than 3
    X3 = random.expovariate(1) # we make an initial sample and then see if we can accept it
    while(not 3*X3 > (15-(X1+2*X2))):
        X3 = random.expovariate(1) # we keep on sampling unless we achieve our desired condition.
    X3_list.append(X3)
    X_sum_list.append(X1 + 2*X2 + 3*X3)
    X2 = random.expovariate(1)
    while(not 2*X2 > (15-(X1+3*X3))):
        X2 = random.expovariate(1)
    X2_list.append(X2)
    X_sum_list.append(X1 + 2*X2 + 3*X3)
    X1 = random.expovariate(1)
    while(not X1 > (15-(2*X2+3*X3))):
        X1 = random.expovariate(1)
    X1_list.append(X1)
    X_sum_list.append(X1 + 2*X2 + 3*X3)

print('Expected value of the desired sum')
print(sum(X_sum_list)/len(X_sum_list))
plt.plot(X_sum_list)
plt.ylabel('X1 + 2*X2 + 3*X3')
plt.xlabel('samples')
plt.title('GIBBS SAMPLING')
plt.show()