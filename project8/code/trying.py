import numpy as np

def sch_fun(x1,x2): 
    return (418.9829*2 -  x1*np.sin(np.sqrt(np.abs(x1))) -  x2*np.sin(np.sqrt(np.abs(x2))))
print(sch_fun(420.9687,420.9687))


