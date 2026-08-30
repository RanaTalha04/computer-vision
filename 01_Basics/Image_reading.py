# Version Checking

import cv2
import numpy as np


print(cv2.__version__)
print(np.__version__)


# Image Reading

image = cv2.imread("../images/cat1.jpg")

print(type(image))
print(image.shape)

cv2.imshow("Cat Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
