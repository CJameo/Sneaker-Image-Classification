for file in jordan1Images:
    try:
        img = loadGray2("Jordan1/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue

for file in react87Images:
    try:
        img = loadGray2("React87/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue

for file in jordan11Images:
    try:
        img = loadGray2("Jordan11/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue
    
for file in am97Images:
    try:
        img = loadGray2("AirMax97/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue
    
for file in shellImages:
    try:
        img = loadGray2("AdidasSuperstarShell/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue
        

for file in y350Images:
    try:
        img = loadGray2("Yeezy350v2/{}".format(file))
        kp, desc = sift.detectAndCompute(img, None)
        BoW.add(desc)
    except:
        continue
    
    
vocabulary = BoW.cluster()
descriptors = BoW.getDescriptors()