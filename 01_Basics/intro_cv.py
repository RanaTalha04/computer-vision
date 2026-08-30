# Version Checking

import cv2
import numpy as np


# print(cv2.__version__)
# print(np.__version__)


# Image Reading

image = cv2.imread("../images/cat1.jpg")

# print(type(image))
# print(image.shape)

# cv2.imshow("Cat Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Image dimension

# height, width, channel = image.shape
# print(f"Height: {height}, Width: {width}, Color Channel: {channel}")

# Accessing pixels

# pixel = image[100, 100]
# print(pixel)

# # Image Croping

# crop = image[100:200, 200: 400]
# print(crop)

# Resizing

# resized = cv2.resize(image, (650, 450))
# # 650 ---> width and 450 ---> height

# cv2.imshow("Reized_image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Gray-scale Conversion

grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_gray_image = cv2.resize(grayed_image, (650, 450))

print(grayed_image.shape)

cv2.imshow("Gray Resized Image", resized_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("../images/graycat.jpg", resized_gray_image)