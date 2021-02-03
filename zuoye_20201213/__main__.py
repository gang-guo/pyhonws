# def rd(data, ndigits=0):
#     if(not isinstance(data, list)):
#         return data
#     elif(data == None or len(data) == 0):
#         return data
#     elif(isinstance(data[0], list)):
#         return [rd(datum, ndigits) for datum in data]
#     else:
#         return [round(datum, ndigits) for datum in data]
#
# print(rd([[1.23,1.5],[1.357]], 1))

# import numpy as np
# arr = np.array([[5.4, 1.533, 1.268], [5.415, 1.509, 1.222]])
# np.round_(arr, 2)
# print(arr)

# import torch
# arr = torch.FloatTensor([[5.4, 1.533, 1.268], [5.415, 1.509, 1.222]])
# print(torch.round(arr * (10 **2)) / (10 **2))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()

ax = Axes3D(fig)

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

x,y = np.meshgrid(x, y)
z = np.abs(x) + np.abs(y)
ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contourf(x,y,z,zdir='z',offset=0)
ax.set_zlim(0, 10)
plt.show()
# print('over')
# print(round(0.305,2))

# print(round(3.315,2))
# print(round(3.325,2))
# print(round(3.335,2))
# print(round(3.345,2))
# print(round(3.355,2))
# print(round(3.365,2))
# print(round(3.375,2))
# print(round(3.385,2))
# print(round(3.395,2))
#
# print(round(3.205,2))
# print(round(3.215,2))
# print(round(3.225,2))
# print(round(3.235,2))
# print(round(3.245,2))
# print(round(3.255,2))
# print(round(3.265,2))
# print(round(3.275,2))
# print(round(3.285,2))
# print(round(3.295,2))

