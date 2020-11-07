import cv2
import os

path = "../images/train"
orb = cv2.ORB_create(nfeatures=1000)

### importing images dataset
images = []
classNames = []
productsList = os.listdir(path)
print(productsList)


for productClass in productsList:
    imgCur = cv2.imread(f'{path}/{productClass}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(productClass)[0])

print(classNames)

def findDes(image):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findId(img, desList, threshold=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalValue = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            matchList.append(len(good))
    except:
        pass
    # print(matchList)

    if len(matchList) != 0:
        if max(matchList) > threshold:
            finalValue = matchList.index(max(matchList))
    return finalValue

desList = findDes(images)

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    id = findId(img2, desList)

    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('img2', img2)
    cv2.waitKey(1)

