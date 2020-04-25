import os
import sys
import pickle

sys.path.append(os.pardir)
from assets.dataset.mnist import load_mnist
import numpy as np
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test


def init_network():
    with open("../assets/dataset/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    guess = np.argmax(y)
    if guess == t[i]:
        # print("correct!")
        accuracy_cnt += 1
    else:
        re_normal = x[i].astype(np.float32)
        re_normal *= 255.0
        img_show(re_normal.reshape(28, 28))
        print(f"wrong! Guess={guess}, ans={t[i]}")
        input("Press button to continue: ")
print(f"accuracy rate is: {accuracy_cnt / len(x)}")
