import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

N_r = 500  #as per the limit of the hyper cube
x = np.linspace(-N_r,N_r,100)
y = np.linspace(-N_r,N_r,100)
X, Y = np.meshgrid(x, y)
z1 = X*np.sin(np.sqrt(np.abs(X)))
z2 = Y*np.sin(np.sqrt(np.abs(Y)))
Z = 418.9829*2 - z1 -z2

plt.figure(num=None,dpi=100)
plt.contourf(X,Y,Z) # plotting the contour
plt.title('Contour Plot')
plt.colorbar()
plt.show()

cplot = plt.figure(num=None,dpi=150)
ax = cplot.add_subplot(111,projection='3d')
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
cbar = cplot.colorbar(surf, shrink=0.5, aspect=5)
cbar.minorticks_on()
plt.show()

def sch_fun(x1,x2): 
    return (418.9829*2 -  x1*np.sin(np.sqrt(np.abs(x1))) -  x2*np.sin(np.sqrt(np.abs(x2))))

# simulated annealing to find the minimum 
N = 10000
x1 = 0 #we begin our simulation at origin
x2 = 0
x1_list = []
x2_list = []
T = 100
value_list = []
value_list.append(sch_fun(x1,x2))

for i in range (1,N):
    x1_prop = np.random.normal(x1,50) # creating a new propsal solution
    x2_prop = np.random.normal(x2,50) 
    alpha = np.exp((sch_fun(x1_prop,x2_prop) - sch_fun(x1,x2))/T) #the threshold value 
    if (sch_fun(x1_prop,x2_prop) > sch_fun(x1,x2)): # cheking the coondition
        x1 = x1_prop #updating new va
        x2 = x2_prop
    else:
        if (np.random.rand() < alpha): #now checking with the alpha, where we sometimes accpet bad values
            x1 = x1_prop 
            x2 = x2_prop

    #T = 100/(np.log(i+2)) #logarithmic cooling function
    #T = 100 - (100/N)*(i+1) #polynomial cooling function
    T = 100*np.exp(-(100/N)*np.sqrt(i+1)) #exponenential cooling function
    x1_list.append(x1)
    x2_list.append(x2)
    value_list.append(sch_fun(x1,x2))

plt.hist(value_list)
plt.show()
plt.plot(x1_list, x2_list, 'C3', zorder=1, lw=3)
plt.scatter(x1_list, x2_list, s=1, zorder=2)
plt.contour(X,Y,Z)
plt.title('final path')
plt.tight_layout()
plt.show()