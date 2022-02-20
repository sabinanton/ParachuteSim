import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt('loadcellstratosiv.txt', skip_header=1).T
data[0] = data[0]
data[1] = data[1] / 10**6
data[0] = -np.where(np.abs(data[0]) > 10000, 0, data[0])
data[1] = np.where(np.abs(data[1]) < 0, 0, data[1])
offset = 10000
plt.plot(data[1][((data[1] > 10.5) & (data[1] < 15.5))], data[0][((data[1] > 10.5) & (data[1] < 15.5))])
plt.grid()
plt.show()