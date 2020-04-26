import numpy as np


def f(x):
    return np.sum(x ** 2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # 丸め誤差が出るよくわからん
    # x = x.astype(dtype="float64")

    # flattenしてidxに入れてくれる(あくまで1次元配列or1m行列)
    for idx in range(x.size):
        x_i = x[idx]

        x[idx] = x_i + h
        f1 = f(x)

        x[idx] = x_i - h
        f2 = f(x)

        grad[idx] = (f1 - f2) / (2 * h)
        x[idx] = x_i

    return grad


def gradient_descent(f, init_x, lr=0.1, step_num=1000):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)

        x -= lr * grad

    return x


# numerical_gradient("",np.array([[0,1],[2,3]]))
q = np.array([-3.0, 4.0])
print(gradient_descent(f, q))
