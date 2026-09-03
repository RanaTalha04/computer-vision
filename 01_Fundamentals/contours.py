import cv2

image = cv2.imread("../images/girl_horse.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"Number of shapes found: {len(contours)}")

output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

if contours: 
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    x, y, w, h = cv2.boundingRect(biggest)
    print(f"Area of biggest shape: {area}")
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
window_width = 800
window_height = 600

windows = {
    "Original Image: ": image,
    "Binary Threshold: ": binary,
    "Contours: ": output
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, window_width, window_height)
    cv2.imshow(win_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()