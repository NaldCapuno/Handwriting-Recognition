import numpy as np
import cv2
import os

Resize_Image_Width = 30
Resize_Image_Height = 40
char_values = list(range(48, 58))
image_dirs = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

flattenedImages = np.empty((0, Resize_Image_Width * Resize_Image_Height))
intClassifications = []

for char, directory in zip(char_values, image_dirs):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath)

        if image is None:
            print(f"Error: Could not load image {filepath}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, imgThresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        imgContours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in imgContours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                imgROI = imgThresh[y:y + h, x:x + w]
                imgResized = cv2.resize(imgROI, (Resize_Image_Width, Resize_Image_Height))

                intClassifications.append(char)
                flattenedImage = imgResized.reshape((1, Resize_Image_Width * Resize_Image_Height))
                flattenedImages = np.append(flattenedImages, flattenedImage, axis=0)

fltClassifications = np.array(intClassifications, np.float32).reshape((-1, 1))

np.savetxt("Classifications.txt", fltClassifications)
np.savetxt("FlattenedImages.txt", flattenedImages)

print("Training Complete")
