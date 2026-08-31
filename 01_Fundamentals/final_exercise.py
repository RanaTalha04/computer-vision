import cv2

# Read
image = cv2.imread("../images/building.jpg")
print(type(image))
print(image.shape)

# Value at pixel (100, 100)
pixel = image[100, 100]
print(pixel)

# Resize 
resized_image = cv2.resize(image, (500, 500)) # first 500 --> width, second 500 --> height

# Crop
cropped_image = resized_image[0:200,  0: 200]
print(cropped_image.shape)

# Convert to Gray Scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_gray_image = cv2.resize(gray_image, (650, 450))

# Crop image converted to Gray Scale
gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
resized_cropped_image = cv2.resize(gray_cropped_image, (650, 450))

# Display
cv2.imshow("Original Image resized: ", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Cropped Gray Scale Image: ", resized_cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Gray Scale Image: ", resized_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
