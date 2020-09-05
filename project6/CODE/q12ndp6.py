#-- POLA MARSAGLIA METHOD----#
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
i = 0
a_list = []
x_list = []
y_list = []
start = timeit.default_timer() # START TIME
while(i<=999999):
    #----- THE ALGORITH-------
    u3 = 2*np.random.rand()-1
    u4 = 2*np.random.rand()-1
    s = u3*u3 + u4*u4
    if(s < 1):
        i = i + 1
        X = math.sqrt(-2*math.log(s)/s)*u3
        x = math.sqrt(V1)*X + M1
        Y = math.sqrt(-2*math.log(s)/s)*u4
        y = math.sqrt(V2)*Y + M2
        x_list.append(x)
        y_list.append(y)
        a = x + y
        a_list.append(a)
stop = timeit.default_timer()
print('Time: ', stop - start)  # STOP TIME
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
plt.title('POLAR MARSAGALIA METHOD')
plt.show()

print('--PRACTICAL VALUES----')
sample_mean_practical = statistics.mean(a_list)
print('sample_mean_practical:',sample_mean_practical)
sample_variance_practical = statistics.variance(a_list)
print('sample_variance_practical:',sample_variance_practical)