import cv2
import numpy as np

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40

charClassifications = np.loadtxt("classifications.txt", np.float32)
flatCharImages = np.loadtxt("flatCharImages.txt", np.float32)
charClassifications = charClassifications.reshape((charClassifications.size, 1))

knn = cv2.ml.KNearest_create()
kNearest = knn
knn.train(flatCharImages, cv2.ml.ROW_SAMPLE, charClassifications)

imgTestSample = cv2.imread("test_image/001.png")
bf = cv2.bilateralFilter(imgTestSample, 50, 100, 100)

imgGray = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
retVal, th = cv2.threshold(imgGray, 150, 255, cv2.CHAIN_APPROX_NONE)
thCopy = th.copy()

contours, h = cv2.findContours(thCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursCopy = contours.copy()

trFinalString = ""

for c in contoursCopy:
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        if len(str(cv2.contourArea(approx))) < 6:
            pass
        else:
            [intX, intY, intW, intH] = cv2.boundingRect(approx)
            cv2.rectangle(imgTestSample, (intX, intY), (intX + intW, intY + intH), (0, 255, 0), 2)

            imgChar = th[intY: intY+intH, intX: intX+intW]
            cv2.imshow('121123123', imgChar)
            cv2.waitKey(0)