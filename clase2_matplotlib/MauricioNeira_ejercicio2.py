import numpy as np  
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

a = np.random.random((4,8))
a[:,-1] = -1
a[1,:] = 2
print(a)

a=np.random.normal(size=1000)
print(len(a[a>2.0]))

a=np.random.normal(size=1000)
b = a
b[b<0] = -1
b[b>=0] = 1
print(b)

import numpy as np
import matplotlib.pyplot as plt
a=5
b=4
delta = np.pi/2
t = np.linspace(0,10,1000)
x = np.sin(a*t+delta)
y = np.sin(b*t)
plt.plot(x,y)
plt.show()
plt.close()

def circlePoints(N):
    r = np.random.random(N)**0.5
    phi = np.random.random(N)*2*np.pi
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x,y
x,y = circlePoints(1000)
plt.scatter(x,y)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
plt.close()


def spherePoints(N):
#     phi = np.random.random(N)*np.pi*2
    theta = np.random.random(N)*np.pi*2
    phi = np.arccos(2.0*np.random.random(N)-1.0);
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return x,y,z

x,y,z = spherePoints(1000)
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')





ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
plt.close()