import numpy as np
import cv2
import math
import pandas as pd
import PIL
import skimage
from sklearn.cluster import dbscan, KMeans
import matplotlib .pyplot as plt

#
def loadGray(img):
    image = cv2.imread(img, 1)
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return image

def displayImg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

# Testing image feature extraction on single image
jordan1 = loadGray("jordan1bred.jpg")
jordan1a = loadGray("jordan1natural.jpg")
jordan1nat = loadGray("jordan1roy_2.jpg")
react87nat = loadGray("react87og_4.jpg")
react87 = loadGray("react87og_5.jpg")
jordan1royalnat = loadGray("1 (1).jpg")
jordan1royalnat = loadGray("1 (3).jpg")


plt.hist(jordan1royalnat[:,:, 0])
plt.show()
plt.hist(jordan1royalnat[:,:, 1])
plt.show()
plt.hist(jordan1royalnat[:,:, 2])
plt.show()

plt.hist(jordan1royalnat[:,:, 0])
plt.show()
plt.hist(jordan1royalnat[:,:, 1])
plt.show()
plt.hist(jordan1royalnat[:,:, 2])
plt.show()