import numpy as np
import random
import math
t = 0       #time variable
Na = 0      #number of arrival
Nd = 0      #number of departure
n = 0       #total jobs in the system
ta = 0      #time of arrival        
i = 0
wait1 = 0
wait_total1 = 0
wait2 = 0
wait_total2 = 0
td = math.inf   #time of departure
T_total = 100
Tp = 0
Ts = np.zeros(T_total*T_total)
lamda_max = 19  #maximum arrival rate
lamda = [4,7,10,13,16,19,16,13,10,7,4]  #list of rate of arrivals
while(Ts[i]<T_total):
    if(ta <= td and ta < T_total): #customer arrives and the queue is open for customers to arrive 
        t = ta
        Na += 1 
        n += 1
        while(t<T_total):
            u1 = np.random.rand()
            t = Ts[i] - np.log(u1)/lamda_max   
            u2 = np.random.rand()
            if (u2 <= lamda[int(np.mod(Ts[i],11))]/lamda_max):
                ta = t
                break
        if(n==1):
            Y = random.expovariate(25)
            td = t + Y
    if(td < ta and td <= T_total): #customer departs and the queue is open for customers to arrive
        t = td
        n = n - 1
        Nd = Nd + 1
        if(n == 0): #if there is no one in the queue then the server goes to sleep
            td = math.inf
            wait1 = random.uniform(0, 0.3)
            t = t + wait1 #the time has to be advanced by wait time
            wait_total1 = wait_total1 + wait1
        else:
            Y = random.expovariate(25)
            td = t + Y    
    if(min(ta,td)>T_total and n>0): #the queue is closed the already existing customers remains queued
        t = td
        n = n-1
        Nd = Nd + 1
        if n==0:
            Y = random.expovariate(25)
            td = t + Y
            wait2 = random.uniform(0, 0.3)
            t = t + wait2
            wait_total2 = wait_total2 + wait2
    if(min(ta,td)>T_total and n==0): #the queue is closed and there is no one in the queue, no new addition is alowed.
        Tp = max(t-T_total,0)  
    Ts[i+1] = t
    i += 1
print('server on break')    
print(wait_total1 + wait_total2)


