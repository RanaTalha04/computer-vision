import cv2
import numpy as np

image = cv2.imread("../images/cat2.jpg")
(h, w) = image.shape[:2]

# 1. Flipping
flip_horizontal = cv2.flip(image, 1)   # 1 = flip left-right
flip_vertical   = cv2.flip(image, 0)   # 0 = flip up-down

# 2. Translation (shifting the image)
M_translate = np.float32([[1, 0, 50], [0, 1, 30]])  # shift 50px right, 30px down
translated = cv2.warpAffine(image, M_translate, (w, h))

# 3. Rotation
center = (w // 2, h // 2)
M_rotate = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(image, M_rotate, (w, h))

# 4. Perspective transform (e.g., straightening a tilted document)
# You give it 4 source points (corners of the tilted area)
# and 4 destination points (where you want those corners to end up)
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(image, M_perspective, (300, 300))

window_width = 800
window_height = 600

windows = {
    "Original": image,
    "Horizontal Flipped": flip_horizontal,
    "Vertical Flipped": flip_vertical,
    "Translated": translated,
    "Rotated": rotated,
    "Perspective Warped": warped
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window_width, window_height)
    cv2.imshow(win_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()