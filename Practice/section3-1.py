import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


x = np.arange(-10, 10, 0.1)
y = ReLU(x)

plt.plot(x, y)
plt.show()

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[1, 2, 3], [0, 1, 1]])
print(A.shape)
print(B.shape)
print(np.dot(A, B))
