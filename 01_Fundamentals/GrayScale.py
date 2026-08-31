import cv2
import numpy as np

image = cv2.imread("../images/forest.jpg")

# Gray-scale Conversion

grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_gray_image = cv2.resize(grayed_image, (650, 450))

print(grayed_image.shape)

cv2.imshow("Gray Resized Image", resized_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("../images/graycat.jpg", resized_gray_image)