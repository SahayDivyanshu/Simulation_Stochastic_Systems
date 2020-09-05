import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



x, y = np.mgrid[0:5:.01, 0:100:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
plt.contourf(x, y, rv.pdf(pos))
plt.show()