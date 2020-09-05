import random 
import math
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt
import statistics 
c_list = []
sample_data = []
i=0
j=0
accept = 0
reject = 0
def pdfx(x): # the gamma pdf
    p = 32/(945*math.sqrt(math.pi))
    q = x**(9/2)
    return p*q*math.exp(-x)

def pdfy(y): # the exponenetial pdf
    return (2/11)*math.exp(-2*y/11)

while(i<10.0): #generating C
    c = pdfx(i)/pdfy(i)
    c_list.append(c)
    i+=0.01
C = max(c_list)  

for j in range (1000):
    Y = random.expovariate(2/11)
    u = np.random.rand()
    value_test = C*u
    if value_test <= pdfx(Y)/pdfy(Y): #the accept reject algorithm
        sample_data.append(Y)
        accept += 1
    else:
        reject += 1

x= np.arange(0,15)
fig = plt.figure()
plt.hist(sample_data)
plt.plot(550*(stats.gamma.pdf(x,5.5,1)))
plt.show()
print('--ACCEPTANCE RATE---')
rate = accept/ (accept + reject)
print(rate)