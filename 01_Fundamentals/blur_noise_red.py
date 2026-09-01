import cv2

image = cv2.imread("../images/building.jpg")

# Simple averaging blur
# the kernel size can be both odd or even number (width, height)

avg_blur = cv2.blur(image, (35, 35)) 

# Guassian Blur - More natural 
# the kernel size should be odd number (width, height)

gaus_blur = cv2.GaussianBlur(image, (35, 35), 0) # 0 --> this value let's opencv auto calculates the blur intesity

# Median Blur - great for removing "salt and pepper" noise (random black/white specks)
# the kernel size should be odd number (img, int)

med_blur = cv2.medianBlur(image, 35)

#  Bilateral filter - blurs while keeping edges sharp (slower, but smart)
# 9 --> Pixel diameter (keep low), the first 75 --> color tolerance (keep low), the second 75 --> coordinate distance tolerance

bilateral = cv2.bilateralFilter(image, 9, 75, 75)

wind_width = 700
wind_height = 500

windows = {
    "Original Image: ": image,
    "Average Blured: ": avg_blur,
    "Gaussian Blur: ": gaus_blur,
    "Median Blur": med_blur,
    "Bilateral Filter: ": bilateral
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, wind_width, wind_height)
    cv2.imshow(win_name, img)


cv2.waitKey(0)
cv2.destroyAllWindows()