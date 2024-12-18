import cv2
import numpy as np

RESIZED_IMAGE_WIDTH = 90
RESIZED_IMAGE_HEIGHT = 140

charClassifications = np.loadtxt("classifications.txt", np.float32)
flatCharImages = np.loadtxt("flatCharImages.txt", np.float32)
charClassifications = charClassifications.reshape((charClassifications.size, 1))

knn = cv2.ml.KNearest_create()
knn.train(flatCharImages, cv2.ml.ROW_SAMPLE, charClassifications)

while True:
    imgTestname = input("Enter test image name: ")
    imgTestSample = cv2.imread(f"test/{imgTestname}")
    if imgTestSample is None:
        print("Error: Test image not found.")
        exit()

    bf = cv2.bilateralFilter(imgTestSample, 50, 100, 100)
    imgGray = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY_INV)
    thCopy = th.copy()

    contours, _ = cv2.findContours(thCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    strFinalString = ""

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            if len(str(cv2.contourArea(approx))) < 6:
                pass
            else:
                [intX, intY, IntW, intH] = cv2.boundingRect(approx)
                cv2.rectangle(imgTestSample, (intX, intY), (intX + IntW, intY + intH), (0, 255, 0), 3)
                imgchar = th[intY:intY + intH, intX:intX + IntW]
                imgchar = imgchar[5:130, 5:80]

                img_inverted = cv2.bitwise_not(imgchar)
                ret, th1 = cv2.threshold(img_inverted, 150, 255, cv2.CHAIN_APPROX_NONE)
                cntr, h = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for myc in cntr:
                    [intX, intY, IntW, intH] = cv2.boundingRect(myc)
                    imgROI = th1[intY:intY + intH, intX:intX + IntW]
                    imgROIres = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

                    cv2.imshow("Results", imgROIres)
                    cv2.waitKey(0)

                    imgROIFinalres = imgROIres.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    imgROIFinalres = np.float32(imgROIFinalres)

                    retVal, result, resp, dist = knn.findNearest(imgROIFinalres, k=1)
                    finalstring = str(chr(int(result[0][0])))
                    strFinalString = strFinalString + finalstring

    print("Recognized String: ", strFinalString)
    cv2.imshow("Processed Image", imgTestSample)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    f = open("results.txt", "w")
    f.write(strFinalString)
    f.close