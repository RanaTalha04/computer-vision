import cv2

# Read
image = cv2.imread("../images/building.jpg")
print(type(image))
print(image.shape)

# Convert BRG -> RGB for matplotlib only
# image_rgb = cv2.cvtColor(image, cv2.COLORBGR2RGB)

# Resize 
resized_image = cv2.resize(image, (650, 450)) # 650 --> width, 450 --> height

# Crop
cropped_image = resized_image[100:400,  150: 500]
print(cropped_image)

# Convert to Gray Scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_gray_image = cv2.resize(gray_image, (650, 450))

# Display
cv2.imshow("Original Image resized: ", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Gray Scale Image: ", resized_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()