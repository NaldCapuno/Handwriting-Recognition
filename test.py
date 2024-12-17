import cv2
import numpy as np

rezimgw = 30 #resize
rezimgh = 40 #resize

#initializiting the training data
charClassification = np.loadtxt("classifications.txt", np.float32)
flatCharImages = np.loadtxt("flatCharImages.txt", np.float32)
charClassification = charClassification.reshape(charClassification.size, 1)

#setting KNN ML tech
knn = cv2.ml.KNearest_create()
KNearest = knn
knn.train(flatCharImages,cv2.ml.ROW_SAMPLE, charClassification)

#Test Image Input
imgtestsample = cv2.imread("test/tests.jpg")
bilateral_filter = cv2.bilateralFilter(imgtestsample,50 , 100, 100)

#RGB2Gray. Treshold2Contour
img_gray = cv2.cvtColor(bilateral_filter,cv2.COLOR_BGR2GRAY)
retval, thresh = cv2.threshold(img_gray, 150, 255, cv2.CHAIN_APPROX_NONE)
threshcopy = thresh.copy()

contours, h = cv2.findContours(threshcopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
strfinal = ""

#Main Algo basically ROI
for c in contours:
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        if len(str(cv2.contourArea(approx))) < 6:
            pass
        else: 
            [intX,intY,IntW,intH] = cv2.boundingRect(approx)
            cv2.rectangle(imgtestsample, (intX,intY) ,(intX+IntW, intY+intH), (0,255,0),2)
            imgchar = thresh[intY:intY+intH, intX:intX+IntW] #crop the box
            imgchar = imgchar [5:120, 10:80] #Crop the border stuff
            
            img_inverted = cv2.bitwise_not(imgchar)
            ret,th1 = cv2.threshold(img_inverted, 150, 255, cv2.CHAIN_APPROX_NONE)
            cntr, h = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for myc in cntr:
                [intX,intY,IntW,intH] = cv2.boundingRect(myc)
                imgROI = th1[intY:intY+intH, intX:intX+IntW]
                imgROIrez = cv2.resize(imgROI, (rezimgw, rezimgh))
                
                cv2.imshow("Testing Shit", imgROIrez)
                cv2.waitKey(0)
                
                Finalrez = imgROIrez.reshape((1, rezimgw * rezimgh))
                Finalrez = np.float32(Finalrez)
                
                retVal, result, resp, dist = KNearest.findNearest(Finalrez, k=1)
                finalstring = str(chr(int(result[0][0])))
                strfinal = strfinal + finalstring
print(strfinal)