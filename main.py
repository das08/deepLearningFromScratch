import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('assets/img/myDAQ_oscillo.png')
plt.imshow(img)

plt.show()