import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../images/mountains2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Pixle brightness")
plt.ylabel("Number of pixel")
plt.show()

# Histogram for a color image; do it per channels
colors = ['b', 'g', 'r']

for i, col in enumerate(colors):
    hist_color = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist_color, color=col)
plt.title("Color Histogram")
plt.show()

# Improve Contrast uing Histogram Equilizer

equalized = cv2.equalizeHist(gray)
cv2.imshow("Before", gray)
cv2.imshow("After Equalization", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()