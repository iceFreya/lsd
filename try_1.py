import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
y = np.array([3.1,5.0,6.9,8.8,10.9]).reshape(-1,1)

w = np.array([[0,0],[0,0]])
Ir = 0.01
epochs = 1000
loss_history = []

for epoch in range(epochs):
    y_hat = np.dot(X,w)

    loss = np.mean((y_hat-y)**2)
    loss_history.append(loss)

    gradient = (2 / len(X)) * np.dot(X.T,(y_hat - y))

    w = w - Ir * gradient

    if epoch % 100 == 0:
        print(f"epoch{epoch},loss{loss},w0{w[0][0]:.4f},w1{w[1][0]:.4f}")

print("outcome")
print(f"b:{w[0][0]:.4f}")
print(f"k:{w[1][0]:.4f}")

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.scatter(X[:,1],y,color='red',label='true')
plt.plot(X[:,1],np.dot(X,w),color='blue',label='line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('outcome')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss_history,color='green')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('process')

plt.tight_layout()
plt.show()