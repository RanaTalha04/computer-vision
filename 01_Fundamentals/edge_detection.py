import cv2

image = cv2.imread("../images/building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Best practice: blur first to reduce noise, so edges are cleaner

blurred = cv2.GaussianBlur(gray, (27, 27), 0)

# Sobel - detects edges in a specific direction (X or Y)

sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) # Vertival Edges
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3) # Horizontal Edges

abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

combined_sobel = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# Laplacian - detects edges in all directions at once

laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Canny - the most popular, well-balanced, industry-standard method

edges = cv2.Canny(gray, threshold1=50, threshold2=150)

wind_width = 700
wind_height = 500

windows = {
    "Original Image: ": image,
    "Gray Image: ": gray,
    "Combined Sobel Method: ": combined_sobel,
    "Laplacian Method: ": laplacian,
    "Canny Edges: ": edges
    
}

for win_name, img in windows.items():
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, wind_width, wind_height)
    cv2.imshow(win_name, img)


cv2.waitKey(0)
cv2.destroyAllWindows()