import random
import numpy as np 
import scipy.stats as stats 
import matplotlib.pyplot as plt
import statistics 
import math
p = 0 #probability of arrival, to be varied over time
r1 = 0.5 #probability that the a packet is switched to 1st port
r2 = 0.5 #probability that the a packet is switched to 1st port
T_Total = 100 #number of times repeating the simulation 
outlist1 = []
outlist2 = []
avg_list = []
packet_process_list = []
efficiency_list = []
packet_process_list_mean = []
prob_arrival = []
index1 = 0
index2 = 0
out_1_to_2 = []
out_1_to_1 = []
out_2_to_2 = []
out_2_to_1 = []

for j in range(T_Total):
    p = np.random.rand()
    prob_arrival.append(p)
    bufferlist1 = [] #input buffer for 1st input 
    bufferlist2 = [] #input buffer for 2nd input

    index1 = 0
    index2 = 0
    packet_process = 0
    prob1 = np.random.rand(T_Total)
    prob2 = np.random.rand(T_Total)
    for i in range(T_Total):
        # ------- ANALYSING PACKET ARRIVAL AT INPUT 1------------------
        #checking if in the previous iteration the packet was not transferred from input 1
        #print('aya_01')
        if(prob1[i] > p): #checking if the packet has arrived at input 1
            bufferlist1.insert(i,1) #updatinig the bufferlist if arrived
            transfer1 = np.random.rand() 
            if(transfer1 < r1): #checking where the packet wants to go
                out_1_to_2.insert(i,1)  #packet from 1 wants to go to 2
                out_1_to_1.insert(i,0)
            else:
                out_1_to_1.insert(i,1)  #packet from 1 wants to go to 1
                out_1_to_2.insert(i,0)
        else:
            bufferlist1.insert(i,0) #updatinig the bufferlist if not arrived
            out_1_to_1.insert(i,0)
            out_1_to_2.insert(i,0)

        #----------- DOING THE SAME FOR INPUT 2 ---------------------
        #checking if in the previous iteration the packet was not transferred from input 2
        
        if(prob2[i] > p): #checking if the packet has arrived at input 1
            bufferlist2.insert(i,1) #updatinig the bufferlist if arrived
            transfer2 = np.random.rand() 
            if(transfer2 < r2): #checking where the packet wants to go
                out_2_to_1.insert(i,1)  #packet from 1 wants to go to 2
                out_2_to_2.insert(i,0)
            else:
                out_2_to_2.insert(i,1)  #packet from 1 wants to go to 1
                out_2_to_1.insert(i,0)
        else:
            bufferlist2.insert(i,0) #updatinig the bufferlist if not arrived
            out_2_to_1.insert(i,0)
            out_2_to_2.insert(i,0)
    #print("aaa")
    while(index1 != T_Total and index2 != T_Total):
        #condition where both at 1 and 2 wishes to go to 2

        if bufferlist1[index1] == 0 and bufferlist2[index2] == 0:
            index1+=1
            index2+=1

        elif bufferlist1[index1] == 0 and bufferlist2[index2] == 1:
            if  (out_2_to_1[index2] == 1):
                packet_process +=1
                outlist1.append(1)
            elif (out_2_to_1[index2] == 1):
                packet_process +=1
                outlist2.append(1)
            index2 +=1
            index1 +=1

        elif bufferlist1[index1] == 1 and bufferlist2[index2] == 0:
            if  (out_1_to_2[index1] == 1):                
                packet_process +=1
                outlist2.append(1)
            elif (out_1_to_1[index1] == 1):                
                packet_process +=1
                outlist1.append(1)
            index2 +=1
            index1 +=1


        elif bufferlist1[index1] == 1 and bufferlist2[index2] == 1:

            if (out_1_to_2[index1] == 1 and out_2_to_2[index2] == 1): 
                transfer3 = np.random.rand()
                if(transfer3 >= 0.5): #randomly decide which should go
                    index1 += 1
                    outlist2.append(1)
                else:
                    index2 += 1
                    outlist2.append(1)

                packet_process += 1

            #condition where both at 1 and 2 wishes to go to 1
            elif (out_1_to_1[index1] == 1 and out_2_to_1[index2] == 1):  #condition where both wishes to go to same output port
                transfer4 = np.random.rand()
                if(transfer4 >= 0.5): #randomly decide which should go
                    index1 += 1
                    outlist1.append(1)
                else:
                    index2 += 1
                    outlist1.append(1)
                
                packet_process += 1
            #condition where there is no conflict        
            elif  (out_1_to_2[index1] == 1 and out_2_to_1[index2] == 1):
                index1 +=1
                index2 +=1
                packet_process +=2
                outlist1.append(1)
                outlist2.append(1)

            #condition where there is no conflict
            elif (out_1_to_1[index1] == 1 and out_2_to_2[index2] == 1):
                index1 +=1
                index2 +=1
                packet_process +=2
                outlist2.append(1)
                outlist1.append(1)      
    #till here one of the list is emptied
    #print("hhh")
    while(index1 != T_Total):           #checking if all list1 elements are tranferred
        if  (out_1_to_2[index1] == 1):
            packet_process +=1
            outlist2.append(1)
        elif (out_1_to_1[index1] == 1):
            packet_process +=1
            outlist1.append(1)
        index1 +=1

    while(index2 != T_Total):           #checking if all list1 elements are tranferred
        if  (out_2_to_1[index2] == 1):
            packet_process +=1
            outlist1.append(1)
        elif (out_2_to_1[index2] == 1):
            packet_process +=1
            outlist2.append(1)
        index2 +=1
    # print(packet_process)
    packet_process_list.append(packet_process)
    packet_process_list_mean.append(packet_process/T_Total)
    efficiency = float(1/max(packet_process_list_mean))
    efficiency_list.append(efficiency)
    buff1 = sum(bufferlist1)
    buff2 = sum(bufferlist2)
    avg = (buff1 + buff2)/2
    avg_list.append(avg)
    efficiency = max(packet_process_list_mean)
    
# print(packet_process_list)
li = [i for i in range(1,T_Total+1)]
fig0 = plt.figure()
plt.xlabel('intervals')
plt.ylabel('probabilities')
plt.plot(li,prob_arrival)
plt.show()

fig1 = plt.figure()
plt.bar(li,avg_list)
plt.plot(prob_arrival, 'r-')
plt.xlabel('intervals')
plt.ylabel('mean of number of packets in buffer 1 and 2')
plt.show()

fig2 = plt.figure()
plt.xlabel('intervals')
plt.ylabel('mean of number of processed packets by the HOL switch')
plt.bar(li,packet_process_list_mean)
plt.plot(prob_arrival, 'r-')
plt.show()

#------- 95% CI -------

print(efficiency_list)
eff_sum = sum(efficiency_list)
eff_avg = eff_sum / T_Total
std_dev = statistics.stdev(efficiency_list) 
moe = 1.96*(std_dev/math.sqrt(T_Total))
CI_l = eff_avg - moe
CI_u = eff_avg + moe
print(CI_l)
print(CI_u)



