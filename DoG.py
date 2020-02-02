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
    image = cv2.imread(img, 0)
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

# LoG estimation
oct10 = jordan1royalnat
oct11 = cv2.GaussianBlur(oct10, ksize = (5, 5), sigmaX=0)
oct12 = cv2.GaussianBlur(oct11, ksize = (5, 5), sigmaX=0)
oct13 = cv2.GaussianBlur(oct12, ksize = (5, 5), sigmaX=0)
oct14 = cv2.GaussianBlur(oct13, ksize = (5, 5), sigmaX=0)
oct15 = cv2.GaussianBlur(oct14, ksize = (5, 5), sigmaX=0)

oct20 = cv2.resize(jordan1royalnat, dsize=(oct10.shape[0]//2, oct10.shape[1]//2))
oct21 = cv2.GaussianBlur(oct20, ksize = (5, 5), sigmaX=0)
oct22 = cv2.GaussianBlur(oct21, ksize = (5, 5), sigmaX=0)
oct23 = cv2.GaussianBlur(oct22, ksize = (5, 5), sigmaX=0)
oct24 = cv2.GaussianBlur(oct23, ksize = (5, 5), sigmaX=0)
oct25 = cv2.GaussianBlur(oct24, ksize = (5, 5), sigmaX=0)

oct30 = cv2.resize(jordan1royalnat, dsize=(oct20.shape[0]//2, oct20.shape[1]//2))
oct31 = cv2.GaussianBlur(oct30, ksize = (5, 5), sigmaX=0)
oct32 = cv2.GaussianBlur(oct31, ksize = (5, 5), sigmaX=0)
oct33 = cv2.GaussianBlur(oct32, ksize = (5, 5), sigmaX=0)
oct34 = cv2.GaussianBlur(oct33, ksize = (5, 5), sigmaX=0)
oct35 = cv2.GaussianBlur(oct34, ksize = (5, 5), sigmaX=0)

oct40 = cv2.resize(jordan1royalnat, dsize=(oct30.shape[0]//2, oct30.shape[1]//2))
oct41 = cv2.GaussianBlur(oct40, ksize = (5, 5), sigmaX=0)
oct42 = cv2.GaussianBlur(oct41, ksize = (5, 5), sigmaX=0)
oct43 = cv2.GaussianBlur(oct42, ksize = (5, 5), sigmaX=0)
oct44 = cv2.GaussianBlur(oct43, ksize = (5, 5), sigmaX=0)
oct45 = cv2.GaussianBlur(oct44, ksize = (5, 5), sigmaX=0)

oct50 = cv2.resize(jordan1royalnat, dsize=(oct40.shape[0]//2, oct40.shape[1]//2))
oct51 = cv2.GaussianBlur(oct50, ksize = (5, 5), sigmaX=0)
oct52 = cv2.GaussianBlur(oct51, ksize = (5, 5), sigmaX=0)
oct53 = cv2.GaussianBlur(oct52, ksize = (5, 5), sigmaX=0)
oct54 = cv2.GaussianBlur(oct53, ksize = (5, 5), sigmaX=0)
oct55 = cv2.GaussianBlur(oct54, ksize = (5, 5), sigmaX=0)
cv2.imwrite("oct10.jpg", oct10)
cv2.imwrite("oct101.jpg", (oct10-oct11))
cv2.imwrite("oct112.jpg", (oct11-oct12))
cv2.imwrite("oct123.jpg", (oct12-oct13))
cv2.imwrite("oct134.jpg", (oct13-oct14))
cv2.imwrite("oct145.jpg", (oct14-oct15))

cv2.imwrite("oct30.jpg", oct30)
cv2.imwrite("oct301.jpg", (oct30-oct31))
cv2.imwrite("oct312.jpg", (oct31-oct32))
cv2.imwrite("oct323.jpg", (oct32-oct33))
cv2.imwrite("oct334.jpg", (oct33-oct34))
cv2.imwrite("oct345.jpg", (oct34-oct35))

cv2.imwrite("oct50.jpg", oct50)
cv2.imwrite("oct501.jpg", (oct50-oct51))
cv2.imwrite("oct512.jpg", (oct51-oct52))
cv2.imwrite("oct523.jpg", (oct52-oct53))
cv2.imwrite("oct534.jpg", (oct53-oct54))
cv2.imwrite("oct545.jpg", (oct54-oct55))





displayImg(test1)
displayImg(test2)
displayImg(test3)
displayImg(test4)
displayImg(test5)
test12 = test1-test2
displayImg(test12)
test45 = test4-test5
displayImg(test45)