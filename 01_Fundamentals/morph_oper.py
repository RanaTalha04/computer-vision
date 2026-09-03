import cv2
import numpy as np

image = cv2.imread("../images/girl_horse.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Kernel Size
kernel = np.ones((5, 5), np.uint8)

# Erosion - shrinks white regions
erosion =  cv2.erode(binary, kernel, iterations=1)

# Dilation - grows white regions
dilation = cv2.dilate(binary, kernel, iterations=1)

# Opening - Erosion followed by Dilation
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Cloing - Dilation followed by Eroion
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


window_width = 800
window_height = 600

windows = {
    "Original Image: ": image,
    "Binary Threshold: ": binary,
    "Erosion": erosion,
    "Dilation": dilation,
    "Opening": opening,
    "Closing": closing
    
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window_width, window_height)
    cv2.imshow(win_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()