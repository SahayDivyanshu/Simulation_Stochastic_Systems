import math
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import poisson

from scipy.stats import poisson
import numpy as np 
import matplotlib.pyplot as plt

x= np.arange(100,300)
p = poisson.pmf(x,120)
plt.plot(x, p*2200)

plt.show()