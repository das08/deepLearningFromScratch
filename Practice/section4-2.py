import numpy as np
import os
import sys

sys.path.append(os.pardir)
from assets.dataset.mnist import load_mnist


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    dy = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + dy)) / batch_size


# print(cross_entropy_error(
#     np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]),
#     np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# ))

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=False, normalize=True)

total_train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(total_train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(t_batch.shape)
print(x_batch.shape)
test=np.array([np.arange(10),t_batch])
print(test.shape)