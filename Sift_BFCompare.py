import numpy as np
import cv2
import math
import pandas as pd
import PIL
import skimage

#
def loadGray(img):
    image = cv2.imread(img, 0)
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return image

def displayImg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Testing image feature extraction on single image
jordan1 = loadGray("jordan1bred.jpg")
jordan1a = loadGray("jordan1natural.jpg")
jordan1nat = loadGray("jordan1roy_2.jpg")
react87nat = loadGray("react87og_4.jpg")
react87 = loadGray("react87og_5.jpg")
jordan1royalnat = loadGray("1 (3).jpg")
threeshoes = loadGray("1 (2).jpg")
jordan11 = loadGray("jordan11concord.jpg")

# creating sift operator
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(jordan1, None)
kp2, desc2 = sift.detectAndCompute(jordan1a, None)
kp3, desc3 = sift.detectAndCompute(jordan1nat, None)
kp4, desc4 = sift.detectAndCompute(jordan1royalnat, None)
kp5, desc5 = sift.detectAndCompute(react87, None)
kp6, desc6 = sift.detectAndCompute(threeshoes, None)
kp7, desc7 = sift.detectAndCompute(jordan11, None)

kpImg1 = cv2.drawKeypoints(jordan1, kp1, None)
kpImg2 = cv2.drawKeypoints(jordan1a, kp2, None)
kpImg3 = cv2.drawKeypoints(jordan1nat, kp3, None)
kpImg4 = cv2.drawKeypoints(jordan1royalnat, kp4, None)
kpImg5 = cv2.drawKeypoints(react87, kp5, None)
kpImg6 = cv2.drawKeypoints(threeshoes, kp6, None)
kpImg7 = cv2.drawKeypoints(jordan11, kp7, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches12 = bf.match(desc1, desc2)
matches12 = sorted(matches12, key = lambda x:x.distance)
match_res12 = cv2.drawMatches(kpImg1, kp1, kpImg2, kp2, matches12[:50], None, flags=2)
matches13 = bf.match(desc1, desc3)
matches13 = sorted(matches13, key = lambda x:x.distance)
match_res13 = cv2.drawMatches(kpImg1, kp1, kpImg3, kp3, matches13[:50], None, flags=2)
matches14 = bf.match(desc1, desc4)
matches14 = sorted(matches14, key = lambda x:x.distance)
match_res14 = cv2.drawMatches(kpImg1, kp1, kpImg4, kp4, matches14[:50], None, flags=2)
matches54 = bf.match(desc5, desc4)
matches54 = sorted(matches54, key = lambda x:x.distance)
match_res54 = cv2.drawMatches(kpImg5, kp5, kpImg4, kp4, matches54[:50], None, flags=2)
matches16 = bf.match(desc1, desc6)
matches16 = sorted(matches16, key = lambda x:x.distance)
match_res16 = cv2.drawMatches(kpImg1, kp1, kpImg6, kp6, matches16[:50], None, flags=2)
matches56 = bf.match(desc5, desc6)
matches56 = sorted(matches56, key = lambda x:x.distance)
match_res56 = cv2.drawMatches(kpImg5, kp5, kpImg6, kp6, matches56[:50], None, flags=2)
matches76 = bf.match(desc7, desc6)
matches76 = sorted(matches76, key = lambda x:x.distance)
match_res76 = cv2.drawMatches(kpImg7, kp7, kpImg6, kp6, matches76[:50], None, flags=2)



cv2.imwrite("16Matches.jpg", match_res16)
cv2.imwrite("56Matches.jpg", match_res56)
cv2.imwrite("76Matches.jpg", match_res76)
cv2.imwrite("14Matches.jpg", match_res14)
cv2.imwrite("13Matches.jpg", match_res13)
cv2.imwrite("12Matches.jpg", match_res12)
cv2.imwrite("keypointsAJ1.jpg", kpImg4)
cv2.imwrite("keypointsAJ1stock.jpg", kpImg1)
cv2.imwrite("keypointsreactstock.jpg", kpImg5)


cv2.imshow("Matching result", match_res12)
cv2.waitKey(5000)
cv2.destroyAllWindows()

cv2.imshow("Keypoint Image", kpImg1)
cv2.imshow("Keypoint Image", kpImg3)
cv2.imwrite("testimage.jpg", match_res14)
cv2.imshow("Matching result", match_res13)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("testimage.jpg", match_res54)
