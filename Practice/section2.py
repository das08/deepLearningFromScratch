import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    res = np.sum(x * w) + b
    if res <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    res = np.sum(x * w) + b
    if res <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    res = np.sum(x * w) + b
    if res <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    a = NAND(x1, x2)
    b = OR(x1, x2)
    return AND(a, b)


for i in [0, 1]:
    for j in [0, 1]:
        print(f"i: {i}, j:{j} ", XOR(i, j))
