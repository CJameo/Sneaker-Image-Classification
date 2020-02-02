import numpy as np
import cv2
import math
import pandas as pd
import PIL
import skimage
from sklearn.cluster import dbscan, KMeans
import matplotlib as plt

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

# creating sift operator
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(jordan1, None)
kp2, desc2 = sift.detectAndCompute(jordan1a, None)
kp3, desc3 = sift.detectAndCompute(jordan1nat, None)
kp4, desc4 = sift.detectAndCompute(react87nat, None)
kp5, desc5 = sift.detectAndCompute(react87, None)

kpImg1 = cv2.drawKeypoints(jordan1, kp1, None)
kpImg2 = cv2.drawKeypoints(jordan1a, kp2, None)
kpImg3 = cv2.drawKeypoints(jordan1nat, kp3, None)
kpImg4 = cv2.drawKeypoints(react87nat, kp4, None)
kpImg5 = cv2.drawKeypoints(react87, kp5, None)
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

# building descriptor histograms for each image
featureDB = np.concatenate([desc1, desc2, desc3, desc4, desc5])
BoW = cv2.BOWKMeansTrainer(800)
for desc in featureDB:
    BoW.add(desc)
dictionary = BoW.cluster()

bowDiction = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
bowDiction.compute(jordan1, sift.detect(jordan1, None))
test = BoW.getDescriptors()
test = sift.detectAndCompute(jordan1, None)

# k-means clustering db images
km = cv2.kmeans(featureDB, K = 25)

# k means clustering db images
kmeans = KMeans(n_clusters=8)
test = kmeans.fit(featureDB)


# Dbscan clustering
db = dbscan(X = featureDB)

np.sum(desc1)



cv2.imshow("Keypoint Image", kpImg1)
cv2.imshow("Keypoint Image", kpImg2)
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


def getLBPimage(gray_image):
    '''
    == Input ==
    gray_image  : color image of shape (height, width)

    == Output ==
    imgLBP : LBP converted image of the same shape as
    '''

    ### Step 0: Step 0: Convert an image to grayscale
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3
    for ih in range(0, gray_image.shape[0] - neighboor):
        for iw in range(0, gray_image.shape[1] - neighboor):
            ### Step 1: 3 by 3 pixel
            img = gray_image[ih:ih + neighboor, iw:iw + neighboor]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()
            ### Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector, 4)
            ### Step 3: Decimal: Convert the binary operated values to a digit.
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            imgLBP[ih + 1, iw + 1] = num
    return (imgLBP)

lbptest = getLBPimage(jordan1)
displayImg(lbptest)
cv2.destroyAllWindows()
