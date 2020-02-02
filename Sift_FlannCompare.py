import numpy as np
import cv2
import math
import pandas as pd
import PIL
import skimage

def loadGray2(img):
    image = cv2.imread(img, 0)
    return image

# loading images
jordan1 = loadGray2("Jordan1/jordan1bred.jpg")
jordan1nat = loadGray2("TestImages/1 (3).jpg")
react87 = loadGray2("React87/download (2).jpg")
react87nat = loadGray2("TestImages/react87og_4.jpg")
jodan11 = loadGray2("TestImages/jordan11basecase.jpg")
threeshoes = loadGray2("TestImages/1 (1).jpg")

sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(jordan1, None)
kp2, desc2 = sift.detectAndCompute(jordan1nat, None)
kp3, desc3 = sift.detectAndCompute(react87, None)
kp4, desc4 = sift.detectAndCompute(react87nat, None)
kp5, desc5 = sift.detectAndCompute(jordan11, None)
kp6, desc6 = sift.detectAndCompute(threeshoes, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)



matches12 = flann.knnMatch(desc1,desc2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches12))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches12):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
match_res12 = cv2.drawMatchesKnn(jordan1,kp1,jordan1nat,kp2,matches12,None,**draw_params)
cv2.imwrite("flannMatches/flann12.jpg", match_res12)

matches34 = flann.knnMatch(desc3,desc4,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches34))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches34):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
match_res34 = cv2.drawMatchesKnn(react87,kp3,react87nat,kp4,matches34,None,**draw_params)
cv2.imwrite("flannMatches/flann34.jpg", match_res34)

matches16 = flann.knnMatch(desc1,desc6,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches16))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches16):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
match_res16 = cv2.drawMatchesKnn(jordan1,kp1,threeshoes,kp6,matches16,None,**draw_params)
cv2.imwrite("flannMatches/flann16.jpg", match_res16)

matches36 = flann.knnMatch(desc3,desc6,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches36))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches36):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
match_res36 = cv2.drawMatchesKnn(react87,kp3,threeshoes,kp6,matches36,None,**draw_params)
cv2.imwrite("flannMatches/flann36.jpg", match_res36)

matches56 = flann.knnMatch(desc5,desc6,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches56))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches56):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
match_res56 = cv2.drawMatchesKnn(jordan11,kp5,threeshoes,kp6,matches56,None,**draw_params)
cv2.imwrite("flannMatches/flann56.jpg", match_res56)