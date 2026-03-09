import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-1,1,50)
y = np.linspace(-1,1,50)
X,Y = np.meshgrid(x,y)

part_1 = 2 * (X ** 4) + (Y ** 2) + np.exp(X ** 2)
part_2 = np.random.normal(0,1,X.shape)
Z = part_1 + part_2

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax_surface = ax.plot_surface(X,Y,Z,cmap = 'viridis',alpha=0.8, rstride=1, cstride=1)
fig.colorbar(ax_surface,ax=ax,shrink=0.6, aspect=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'3D Surface Plot of $z=2x^4+y^2+e^{x^2}+\varepsilon$')

plt.show()