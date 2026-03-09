import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 10000
x = np.random.uniform(-1,1,n)
y = np.random.uniform(-1,1,n)
z = 2 * (x ** 4) + (y ** 2) + np.exp(x ** 2) + np.random.normal(0,1,n)

tensor_x = torch.tensor(x,dtype=torch.float32).reshape(-1,1)
tensor_y = torch.tensor(y,dtype=torch.float32).reshape(-1,1)
X = torch.cat((tensor_x,tensor_y),1)
Y = torch.tensor(z,dtype=torch.float32).reshape(-1,1)

model = nn.Sequential(
    nn.Linear(2,32),
    nn.ReLU(),
    nn.Linear(32,1),
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
losses = []
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
print("finish")

plt.figure(figsize=(8,8))
plt.plot(losses,color='blue',label='loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training Loss Change")
plt.grid(alpha=0.5)
plt.show()

grid_size = 50
x_grid = np.linspace(-1,1,grid_size)
y_grid = np.linspace(-1,1,grid_size)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = 2 * (X_grid ** 4) + (Y_grid ** 2) + np.exp(X_grid ** 2) + np.random.normal(0, 1,X_grid.shape )

x_grid_tensor = torch.tensor(X_grid.reshape(-1,1),dtype=torch.float32)
y_grid_tensor = torch.tensor(Y_grid.reshape(-1,1),dtype=torch.float32)
X_grid_input = torch.cat((x_grid_tensor,y_grid_tensor),1)
Z_pred = model(X_grid_input).detach().numpy().reshape(grid_size,grid_size)

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121,projection='3d')
ax1.plot_surface(X_grid,Y_grid,Z_grid,rstride=1, cstride=1, cmap='viridis', alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('True Function Surface')

ax2 = fig.add_subplot(122,projection='3d')
ax2.plot_surface(X_grid,Y_grid,Z_pred,rstride=1, cstride=1, cmap='viridis', alpha=0.3)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Predicted Function Surface')

plt.tight_layout()
plt.show()