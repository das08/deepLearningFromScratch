import sys, os

sys.path.append(os.pardir)
import numpy as np
from assets.common.functions import softmax, cross_entropy_error
from assets.common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net=simpleNet()
x=np.array([0.6,0.9])
p=net.predict(x)
print(np.argmax(p))
print(np.argmax(p, axis=0))
