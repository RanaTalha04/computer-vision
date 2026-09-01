import cv2


image = cv2.imread("../images/mountains1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple Thresholding: 127 pixels becomes white and remaining black

ret, simple_thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
print(f"Simple Threshold value: {ret} pixel white and remaining black.")

# Inverted Thresholding: 127 pixels becomes black and remaining white

ret, inv_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
print(f"Inverted Thresholding value: {ret} pixel black and remaining white.")

# Otsu's Method: Automatically picks the best threshold value for you

ret, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu picked this threshold value automatically: {ret}")

# Adaptive Threshold: Useful when lightning is uneven across the image

adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

window_width = 800
window_height = 600

windows = {
    "Original Image: ": image,
    "Simple Threshold": simple_thres,
    "Inverse Threshold": inv_thresh,
    "Otsu Threshold": otsu_thresh,
    "Adaptive Threshold": adaptive_thresh
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window_width, window_height)
    cv2.imshow(win_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()