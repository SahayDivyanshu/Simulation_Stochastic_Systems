import numpy as np
import math
Z_list = []
Y = [[1],[2],[3]]
print(Y)
print('---CHOLESKY DECOMPOSITION---')
import numpy as np
a = [[3, -1, 1], [-1, 5, 3], [1, 3, 4]]
print("Original array:")
print(a)
A = np.linalg.cholesky(a)
print("Lower-trianglular A in the Cholesky decomposition of the said array:")
print(A)
for i in range(3):
    u1 = np.random.rand() 
    u2 = np.random.rand() 
    Z = (math.sqrt(-2*math.log(u1)))*math.cos(2*math.pi*u2)
    z = math.sqrt(1)*Z + 0
    Z_list.append([z]) 
X = np.dot(A,Z_list) 
result = [[X[i][j] + Y[i][j]  for j in range
(len(X[0]))] for i in range(len(X))]   
c = 1 
for i in result:
    print('x'+ str(c)+':' + str(i)) 
    c +=1
