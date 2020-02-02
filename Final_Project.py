import numpy as np
import cv2
import math
import pandas as pd
import PIL
import skimage
import os
import imghdr
from sklearn.metrics import confusion_matrix
from sklearn import tree 
import sklearn.metrics as perf
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB




def loadGray(img):
    image = cv2.imread(img, 0)
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return image


def loadGray2(img):
    image = cv2.imread(img, 0)
    return image



def displayImg(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Testing image feature extraction on single image
jordan1 = loadGray("Jordan1/jordan1bred.jpg")
jordan1a = loadGray("Jordan1/jordan1natural.jpg")
jordan1nat = loadGray("TestImages/jordan1roy_2.jpg")
react87nat = loadGray("TestImages/react87og_4.jpg")
react87 = loadGray("React87/react87og_5.jpg")
jordan1royalnat = loadGray("TestImages/1 (3).jpg")
threeshoes = loadGray("TestImages/1 (2).jpg")
jordan11 = loadGray("Jordan11/jordan11concord.jpg")

imageDB = [jordan1, jordan1a, jordan1nat, react87nat, react87, jordan1royalnat, jordan11]

# creating sift operator
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(jordan1, None)
kp2, desc2 = sift.detectAndCompute(jordan1a, None)
kp3, desc3 = sift.detectAndCompute(jordan1nat, None)
kp4, desc4 = sift.detectAndCompute(jordan1royalnat, None)
kp5, desc5 = sift.detectAndCompute(react87, None)
kp6, desc6 = sift.detectAndCompute(threeshoes, None)
kp7, desc7 = sift.detectAndCompute(jordan11, None)

# Drawing keypoint images
kpImg1 = cv2.drawKeypoints(jordan1, kp1, None)
kpImg2 = cv2.drawKeypoints(jordan1a, kp2, None)
kpImg3 = cv2.drawKeypoints(jordan1nat, kp3, None)
kpImg4 = cv2.drawKeypoints(jordan1royalnat, kp4, None)
kpImg5 = cv2.drawKeypoints(react87, kp5, None)
kpImg6 = cv2.drawKeypoints(threeshoes, kp6, None)
kpImg7 = cv2.drawKeypoints(jordan11, kp7, None)

# Drawing KP match images
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

# Creating Image DB
jordan1Images = os.listdir("Jordan1")
react87Images = os.listdir("React87")
jordan11Images = os.listdir("Jordan11")
am97Images = os.listdir("AirMax97")
shellImages = os.listdir("AdidasSuperstarShell")
y350Images = os.listdir("Yeezy350v2")

foldernames = ["Jordan1", "React87", "Jordan11", "AirMax97", "AdidasSuperstarShell", "Yeezy350v2"]

folders = {"Jordan1":jordan1Images, "React87":react87Images, "Jordan11":jordan11Images, 
           "AirMax97":am97Images, "AdidasSuperstarShell":shellImages, "Yeezy350v2":y350Images}

# Removing non-image files
filetypes = ['rgb','gif','pbm','pgm','ppm','tiff','rast','xbm','jpeg','bmp','png','webp','exr']

for name, folder in folders.items():
    for file in folder:
        if imghdr.what("{}/{}".format(name, file), h=None) not in filetypes:
            os.remove("{}/{}".format(name, file))

# Creating cleansed image DB
jordan1Images = os.listdir("Jordan1")
react87Images = os.listdir("React87")
jordan11Images = os.listdir("Jordan11")
am97Images = os.listdir("AirMax97")
shellImages = os.listdir("AdidasSuperstarShell")
y350Images = os.listdir("Yeezy350v2")

foldernames = ["Jordan1", "React87", "Jordan11", "AirMax97", "AdidasSuperstarShell", "Yeezy350v2"]

folders = {"Jordan1":jordan1Images, "React87":react87Images, "Jordan11":jordan11Images, 
           "AirMax97":am97Images, "AdidasSuperstarShell":shellImages, "Yeezy350v2":y350Images}

# Creating K-Means Trainer with n clusters and building vocabulary
BoW = cv2.BOWKMeansTrainer(50)

for name, folder in folders.items():
    count = 0
    for file in folder:
        while count < 100:
            try:
                count = count + 1
                img = loadGray2("{}/{}".format(name, file))
                kp, desc = sift.detectAndCompute(img, None)
                BoW.add(desc)
            except:
                continue
    print("Added {}".format(count))

vocabulary = BoW.cluster()
descriptors = BoW.getDescriptors()

# Setting vocabulary for modeling 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
bow_extract = cv2.BOWImgDescriptorExtractor(sift, flann)
bow_extract.setVocabulary(vocabulary)

# Creating training data
labelVals = {"Jordan1":1, "React87":2, "Jordan11":3, "AirMax97":4, "AdidasSuperstarShell":5, "Yeezy350v2":6}
traindata = []
trainlabels = []

for name, folder in folders.items():
    count = 0
    for file in folder:
        while count < 100:
            try:
                count = count + 1
                img = loadGray2("{}/{}".format(name, file))
                siftkp = sift.detect(img)
                bowsig = bow_extract.compute(img, siftkp)
                traindata.extend(bowsig)
                trainlabels.append(labelVals[name])
            except:
                continue
    
# Creating and training SVM using computed clusters
svm = cv2.ml.SVM_create()
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

# Testing on 10 test images for validity of model (2 stock photos 8 natural)
testImages = os.listdir("TestImages")
predictionLabels = {}
predictions = []
actual = [1, 1, 1, 1, 3, 2, 2, 2, 2, 2]
for image in testImages[3:13]:
    img = loadGray2("TestImages/{}".format(image))
    siftkp = sift.detect(img)
    bowsig = bow_extract.compute(img, siftkp)
    pred = svm.predict(bowsig)
    print(pred)
    predictions.append(pred)
    predictionLabels[image] = pred


# Plotting Confusion matrix
n = 1 # N. . .
predicted = [int(x[n][0][0]) for x in predictions]
confusion = confusion_matrix(actual, predicted, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()

accuracy = perf.accuracy_score(actual, predictions)
precision = perf.precision_score(actual, predictions, average="micro")
recall = perf.recall_score(actual, predictions, average="micro")
f1 = perf.f1_score(actual, predictions, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    

# Using a DT model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(traindata, trainlabels)

# Testing on 10 test images for validity of model (2 stock photos 8 natural)
testImages = os.listdir("TestImages")
predictionLabels = {}
predictions = []
actual = [1, 1, 1, 1, 3, 2, 2, 2, 2, 2]
for image in testImages[3:13]:
    img = loadGray2("TestImages/{}".format(image))
    siftkp = sift.detect(img)
    bowsig = bow_extract.compute(img, siftkp)
    pred = clf.predict(bowsig)
    print(pred)
    predictions.append(pred)
    predictionLabels[image] = pred
    
# Plotting Confusion matrix
confusion2 = confusion_matrix(actual, predictions, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion2, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()


accuracy = perf.accuracy_score(actual, predictions)
precision = perf.precision_score(actual, predictions, average="micro")
recall = perf.recall_score(actual, predictions, average="micro")
f1 = perf.f1_score(actual, predictions, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))

    
# Naive bayes
gnb = GaussianNB()
gnb = gnb.fit(traindata, trainlabels)

# Testing on 10 test images for validity of model (2 stock photos 8 natural)
testImages = os.listdir("TestImages")
predictionLabels = {}
predictions = []
actual = [1, 1, 1, 1, 3, 2, 2, 2, 2, 2]
for image in testImages[3:13]:
    img = loadGray2("TestImages/{}".format(image))
    siftkp = sift.detect(img)
    bowsig = bow_extract.compute(img, siftkp)
    pred = gnb.predict(bowsig)
    print(pred)
    predictions.append(pred)
    predictionLabels[image] = pred
    
# Plotting Confusion matrix
confusion3 = confusion_matrix(actual, predictions, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion3, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()

accuracy = perf.accuracy_score(actual, predictions)
precision = perf.precision_score(actual, predictions, average="micro")
recall = perf.recall_score(actual, predictions, average="micro")
f1 = perf.f1_score(actual, predictions, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    


# Creating actual value array for comparison
actualVals = []
testImages2 = os.listdir("TestImages2")
for typ in testImages2:
    folder = os.listdir("TestImages2/{}".format(typ))
    for file in folder:
        actualVals.append(int(typ))
    

# Testing on stock images
predictions = []
count = 0
for typ in testImages2:
    folder = os.listdir("TestImages2/{}".format(typ))
    print(typ)
    for file in folder:
        count = count + 1
        img = loadGray2("TestImages2/{}/{}".format(typ, file))
        siftkp = sift.detect(img)
        bowsig = bow_extract.compute(img, siftkp)
        pred = svm.predict(bowsig)
        print(pred)
        print(count)
        predictions.append(pred) 
  
      
# Plotting Confusion matrix
n = 1 # N. . .
predicted = [int(x[n][0][0]) for x in predictions]
confusion = confusion_matrix(actualVals, predicted, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()

accuracy = perf.accuracy_score(actualVals, predicted)
precision = perf.precision_score(actualVals, predicted, average="micro")
recall = perf.recall_score(actualVals, predicted, average="micro")
f1 = perf.f1_score(actualVals, predicted, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    

# Using a DT model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(traindata, trainlabels)

# Testing on stock
predictions = []
count = 0
for typ in testImages2:
    folder = os.listdir("TestImages2/{}".format(typ))
    print(typ)
    for file in folder:
        count = count + 1
        img = loadGray2("TestImages2/{}/{}".format(typ, file))
        siftkp = sift.detect(img)
        bowsig = bow_extract.compute(img, siftkp)
        pred = clf.predict(bowsig)
        print(pred)
        print(count)
        predictions.append(pred) 
    
# Plotting Confusion matrix
confusion2 = confusion_matrix(actualVals, predictions, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion2, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()


accuracy = perf.accuracy_score(actualVals, predictions)
precision = perf.precision_score(actualVals, predictions, average="micro")
recall = perf.recall_score(actualVals, predictions, average="micro")
f1 = perf.f1_score(actualVals, predictions, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))

    
# Naive bayes
gnb = GaussianNB()
gnb = gnb.fit(traindata, trainlabels)

# Testing on stock
predictions = []
count = 0
for typ in testImages2:
    folder = os.listdir("TestImages2/{}".format(typ))
    print(typ)
    for file in folder:
        count = count + 1
        img = loadGray2("TestImages2/{}/{}".format(typ, file))
        siftkp = sift.detect(img)
        bowsig = bow_extract.compute(img, siftkp)
        pred = gnb.predict(bowsig)
        print(pred)
        print(count)
        predictions.append(pred) 
    
# Plotting Confusion matrix
confusion3 = confusion_matrix(actualVals, predictions, labels=[1, 2, 3, 4, 5, 6])

ax= plt.subplot()
sns.heatmap(confusion3, annot=True, ax = ax); 
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
ax.yaxis.set_ticklabels(["AJ1", "React87", "AJ11", "AM97", "Shell", "YZY350"])
plt.show()

accuracy = perf.accuracy_score(actualVals, predictions)
precision = perf.precision_score(actualVals, predictions, average="micro")
recall = perf.recall_score(actualVals, predictions, average="micro")
f1 = perf.f1_score(actualVals, predictions, average="micro")
print("Accuracy: {} \nPrecision: {}\nRecall: {}\nF1: {}".format(accuracy, precision, recall, f1))
    