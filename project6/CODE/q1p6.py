
#-- BOX MULLER METHOD----#
import random 
import math
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt
import statistics 
import timeit
#--- THEORETICAL VALUES GIVEN
M1 = 1
M2 = 2
V1 = 4                  
V2 = 9
a_list = []
x_list = []
y_list = []
start = timeit.default_timer() # START TIME
for i in range (1000000):
    #----- THE ALGORITH-------
    u1 = np.random.rand()
    u2 = np.random.rand()

    X = (math.sqrt(-2*math.log(u1)))*math.cos(2*math.pi*u2)
    Y = (math.sqrt(-2*math.log(u1)))*math.sin(2*math.pi*u2)

    x = math.sqrt(V1)*X + M1
    x_list.append(x)
    y = math.sqrt(V2)*Y + M2
    y_list.append(y)
    a = x + y
    a_list.append(a)
stop = timeit.default_timer() # STOP TIME
print('Time: ', stop - start) 
x_min = -20.0
x_max = 16.0
mean = 3.0 
std = 3.6
x_plot = np.linspace(x_min, x_max)
y_plot = stats.norm.pdf(x_plot,mean,std)
cov_mat = np.stack((x_list, y_list))
print(np.cov(cov_mat))
fig = plt.figure()
plt.hist(a_list)
plt.plot(x_plot,3042000*y_plot, color='red')
plt.xlabel('Samples Taken')
plt.ylabel('Frquency of Occurance of the Sample')
plt.title('BOX MULLER METHOD')
plt.show()

print('--PRACTICAL VALUES----')
sample_mean_practical = statistics.mean(a_list)
print('sample_mean_practical:',sample_mean_practical)
sample_variance_practical = statistics.variance(a_list)
print('sample_variance_practical:',sample_variance_practical)











