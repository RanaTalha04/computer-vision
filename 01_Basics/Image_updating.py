import cv2
import numpy as np

image = cv2.imread("../images/cat2.jpg")

# Image dimension

height, width, channel = image.shape
print(f"Height: {height}, Width: {width}, Color Channel: {channel}")

# Accessing pixels

pixel = image[100, 100]
print(pixel)

# Image Croping

crop = image[100:200, 200: 400]
print(crop.shape)

# Image Padding

padding = cv2.copyMakeBorder(image, 
                             top=50, bottom=50, left=50, right=50,
                             borderType=cv2.BORDER_CONSTANT,
                             value=[0, 0, 0])

print(padding.shape)

# Resizing and interpolation

resized = cv2.resize(image, (650, 450), interpolation=cv2.INTER_AREA)
# 650 ---> width and 450 ---> height

cv2.imshow("Reized_image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

