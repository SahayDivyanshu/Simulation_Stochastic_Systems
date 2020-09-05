import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.random import sample_without_replacement
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
cities_coord = []
with open('D:\\SHIVAM\\USC_STUDY\\EE511\\project8\\code\\Q4\\xy_cord.txt') as f:
    for line in f:
        total = line.split()
        cities_coord.append([float(total[0]),float(total[1])])
    # x1 = [float(line.split()[0]) for line in f]
print(cities_coord[3])
# echanged the 1st and 4th row to start from sacramanto
N_cities = 48
p_len = 0

for a1 in range(0,N_cities-1):
    p_len = p_len + distance.euclidean(cities_coord[a1],cities_coord[a1+1])
print('Initial path length:',str(p_len))

# lets name this 1st path as path where the cities as given in file are nubered from 0 to 47
path = np.arange(0,N_cities)
num_iter = 10000
# Save the paths and lengths
pathHistory = np.zeros((num_iter,N_cities))
lenHistory = []
thresh_ar = []

# plot cities and initial path
x_coord = []
y_coord = []
plt.figure()
for i in range(len(cities_coord)):
    x_coord.append(cities_coord[i][0])
    y_coord.append(cities_coord[i][1])
   
plt.plot(x_coord, y_coord, 'C3', zorder=1, lw=3)
plt.scatter(x_coord, y_coord, s=120, zorder=2)
plt.title('Initial path')
plt.tight_layout()
plt.show()

# now we will chooswe new paths by swapping two cities selected randomly

iter_count = 1
path_new = []
c = 1
while iter_count < num_iter:
    iter_count = iter_count + 1
    # Create path p2 by randomly swap two cities
    # index of two cities for the new path
    swap_i, swap_j = np.random.choice(N_cities, 2)
    path_new = np.copy(path)
    path_new[swap_i], path_new[swap_j] = path_new[swap_j], path_new[swap_i]
    # initialize new path length
    p_len2 = 0
    for a1 in range(0,N_cities-1):
        p_len2 = p_len2 + distance.euclidean(cities_coord[path_new[a1]],cities_coord[path_new[a1+1]])

    thresh = np.power((1+iter_count),((p_len - p_len2)/c))
    if p_len2 - p_len <= 0:
        path = np.copy(path_new)
        p_len = np.copy(p_len2)
    else:
        if np.random.rand() <= thresh:
            path = np.copy(path_new)
            p_len = np.copy(p_len2)
          # bookeeping
    pathHistory[iter_count-1][0:len(path_new)] = path_new
    lenHistory.append(p_len2)
    thresh_ar.append(thresh)
plt.figure(num=None,dpi=100)
plt.plot(lenHistory)
print('Optimum path length:',str(p_len))
plt.title('Length of path in each iteration')
plt.show()
x_coord_f = []
y_coord_f = []
ind_f = pathHistory[-1,:].astype(int)
print(ind_f)
for i in range(len(ind_f)):
    x_coord_f.append(cities_coord[ind_f[i]][0])
    y_coord_f.append(cities_coord[ind_f[i]][1])
plt.figure(num=None,dpi=100)
plt.title('Final path')
plt.plot(x_coord_f, y_coord_f, '-o')
plt.show()