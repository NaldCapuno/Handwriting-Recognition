import numpy as np
import cv2

Resize_Image_Width = 30
Resize_Image_Height = 40

char0 = 48
char1 = 49
char2 = 50
char3 = 51
char4 = 52
char5 = 53
char6 = 54
char7 = 55
char8 = 56
char9 = 57

imageTraining0 = cv2.imread('Training0.jpg')
imageTraining1 = cv2.imread('Training1.jpg')
imageTraining2 = cv2.imread('Training2.jpg')
imageTraining3 = cv2.imread('Training3.jpg')
imageTraining4 = cv2.imread('Training4.jpg')
imageTraining5 = cv2.imread('Training5.jpg')
imageTraining6 = cv2.imread('Training6.jpg')
imageTraining7 = cv2.imread('Training7.jpg')
imageTraining8 = cv2.imread('Training8.jpg')
imageTraining9 = cv2.imread('Training9.jpg')

imagGray0 = cv2.cvtColor(imageTraining0, cv2.COLOR_BGR2GRAY)
imagGray1 = cv2.cvtColor(imageTraining1, cv2.COLOR_BGR2GRAY)
imagGray2 = cv2.cvtColor(imageTraining2, cv2.COLOR_BGR2GRAY)
imagGray3 = cv2.cvtColor(imageTraining3, cv2.COLOR_BGR2GRAY)
imagGray4 = cv2.cvtColor(imageTraining4, cv2.COLOR_BGR2GRAY)
imagGray5 = cv2.cvtColor(imageTraining5, cv2.COLOR_BGR2GRAY)
imagGray6 = cv2.cvtColor(imageTraining6, cv2.COLOR_BGR2GRAY)
imagGray7 = cv2.cvtColor(imageTraining7, cv2.COLOR_BGR2GRAY)
imagGray8 = cv2.cvtColor(imageTraining8, cv2.COLOR_BGR2GRAY)
imagGray9 = cv2.cvtColor(imageTraining9, cv2.COLOR_BGR2GRAY)

revatal0, imgTresh0 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal1, imgTresh1 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal2, imgTresh2 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal3, imgTresh3 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal4, imgTresh4 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal5, imgTresh5 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal6, imgTresh6 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal7, imgTresh7 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal8, imgTresh8 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)
revatal9, imgTresh9 = cv2.Threshold(imagGray9, 150, 255,cv2.CHAIN_APPROX_NONE)

imageThreshCopy0 = imgTresh0.copy()
imageThreshCopy1 = imgTresh1.copy()
imageThreshCopy2 = imgTresh2.copy()
imageThreshCopy3 = imgTresh3.copy()
imageThreshCopy4 = imgTresh4.copy()
imageThreshCopy5 = imgTresh5.copy()
imageThreshCopy6 = imgTresh6.copy()
imageThreshCopy7 = imgTresh7.copy()
imageThreshCopy8 = imgTresh8.copy()
imageThreshCopy9 = imgTresh9.copy()

imgContours0, h0 = cv2.findContours(imageThreshCopy0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours1, h1 = cv2.findContours(imageThreshCopy1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours2, h2 = cv2.findContours(imageThreshCopy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours3, h3 = cv2.findContours(imageThreshCopy3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours4, h4 = cv2.findContours(imageThreshCopy4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours5, h5 = cv2.findContours(imageThreshCopy5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours6, h6 = cv2.findContours(imageThreshCopy6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours7, h7 = cv2.findContours(imageThreshCopy7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours8, h8 = cv2.findContours(imageThreshCopy8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours9, h9 = cv2.findContours(imageThreshCopy9, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

flattenedImages = np.empty((0, Resize_Image_Width * Resize_Image_Height))
intClassifications = []

for c1 in imgContours0:
    if cv2.contourArea(c1) > 50:
        [x, y, w, h ]= cv2.boundingRect(c1)
        imgROI1 = imgTresh0[y:y + h, x:x + w]
        imgResizedRoi1 = cv2.resize(imgROI1, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char0)
        flattenedImage = imgResizedRoi1.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c2 in imgContours1:
    if cv2.contourArea(c2) > 50:
        [x, y, w, h ]= cv2.boundingRect(c2)
        imgROI2 = imgTresh1[y:y + h, x:x + w]
        imgResizedRoi2 = cv2.resize(imgROI2, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char1)
        flattenedImage = imgResizedRoi2.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c3 in imgContours2:
    if cv2.contourArea(c3) > 50:
        [x, y, w, h ]= cv2.boundingRect(c3)
        imgROI3 = imgTresh2[y:y + h, x:x + w]
        imgResizedRoi3 = cv2.resize(imgROI3, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char2)
        flattenedImage = imgResizedRoi3.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c4 in imgContours3:
    if cv2.contourArea(c4) > 50:
        [x, y, w, h ]= cv2.boundingRect(c4)
        imgROI4 = imgTresh3[y:y + h, x:x + w]
        imgResizedRoi4 = cv2.resize(imgROI4, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char3)
        flattenedImage = imgResizedRoi4.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c5 in imgContours4:
    if cv2.contourArea(c5) > 50:
        [x, y, w, h ]= cv2.boundingRect(c5)
        imgROI5 = imgTresh4[y:y + h, x:x + w]
        imgResizedRoi5 = cv2.resize(imgROI5, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char4)
        flattenedImage = imgResizedRoi5.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c6 in imgContours5:
    if cv2.contourArea(c6) > 50:
        [x, y, w, h ]= cv2.boundingRect(c6)
        imgROI6 = imgTresh5[y:y + h, x:x + w]
        imgResizedRoi6 = cv2.resize(imgROI6, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char5)
        flattenedImage = imgResizedRoi6.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c7 in imgContours6:
    if cv2.contourArea(c7) > 50:
        [x, y, w, h ]= cv2.boundingRect(c7)
        imgROI7 = imgTresh6[y:y + h, x:x + w]
        imgResizedRoi7 = cv2.resize(imgROI7, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char6)
        flattenedImage = imgResizedRoi7.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c8 in imgContours7:
    if cv2.contourArea(c8) > 50:
        [x, y, w, h ]= cv2.boundingRect(c8)
        imgROI8 = imgTresh7[y:y + h, x:x + w]
        imgResizedRoi8 = cv2.resize(imgROI8, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char7)
        flattenedImage = imgResizedRoi8.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

for c9 in imgContours8:
    if cv2.contourArea(c9) > 50:
        [x, y, w, h ]= cv2.boundingRect(c9)
        imgROI9 = imgTresh8[y:y + h, x:x + w]
        imgResizedRoi9 = cv2.resize(imgROI9, (Resize_Image_Width, Resize_Image_Height))

        intClassifications.append(char8)
        flattenedImage = imgResizedRoi9.reshape((1, Resize_Image_Width * Resize_Image_Height))
        flattenedImages = np.append(flattenedImages, flattenedImage, 0)

fltClassifications = np.array(intClassifications, np.float32)
fltClassifications = fltClassifications.reshape((fltClassifications.size, 1))

print("Training Complete")

np.savetxt("Classifications.txt", fltClassifications)
np.savetxt("FlattenedImages.txt", flattenedImages)



