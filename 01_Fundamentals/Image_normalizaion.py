import cv2
import numpy as np

image = cv2.imread("../images/girl_horse.jpg")

# Convert to float (important) -- Int can't hold decimals

image_float = image.astype(np.float32)

# Normalize from [0, 255] to [0, 1]

normalized_image = image_float / 255.0

print(image[100, 100])
print(normalized_image[100, 100])

# Built-in function

norm_img = cv2.normalize(image, None, dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)
cv2.imshow("Normalized Image: ", norm_img)
cv2.waitKey(1)
cv2.destroyAllWindows()