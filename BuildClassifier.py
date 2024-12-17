import numpy as np
import cv2
import os

Resize_Image_Width = 28
Resize_Image_Height = 28

char_digits = list(range(48, 58))

flattenedImages = np.empty((0, Resize_Image_Width * Resize_Image_Height))
intClassifications = []

dataset_folder = "dataset"

for i, char in enumerate(char_digits):
    digit_folder = os.path.join(dataset_folder, str(i), "JPEG")

    if not os.path.exists(digit_folder):
        continue

    image_files = [f for f in os.listdir(digit_folder) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(digit_folder, image_file)
        imageTraining = cv2.imread(image_path)

        if imageTraining is None:
            continue

        imagGray = cv2.cvtColor(imageTraining, cv2.COLOR_BGR2GRAY)
        _, imgThresh = cv2.threshold(imagGray, 150, 255, cv2.THRESH_BINARY_INV)

        imgContours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imgContours = sorted(imgContours, key=cv2.contourArea, reverse=True)

        for contour in imgContours[:1]:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                imgROI = imgThresh[y:y + h, x:x + w]
                imgResizedRoi = cv2.resize(imgROI, (Resize_Image_Width, Resize_Image_Height))

                intClassifications.append(char)
                flattenedImage = imgResizedRoi.reshape((1, Resize_Image_Width * Resize_Image_Height))
                flattenedImages = np.append(flattenedImages, flattenedImage, 0)

fltClassifications = np.array(intClassifications, np.float32)
fltClassifications = fltClassifications.reshape((fltClassifications.size, 1))

print("Training Complete")
np.savetxt("classifications.txt", fltClassifications)
np.savetxt("flatCharImages.txt", flattenedImages)
