import cv2

#  Format 1: (x1, y1, x2, y2) - top-left corner and bottom-right corner
# This is what YOLO's Python API (results[0].boxes.xyxy) gives you by default

box_xyxy = [4, 229, 796, 728] # x1=4, y1=229, x2=796, y2=728

# Format 2: (x_center, y_center, width, height) - center point + size
# This is the format YOLO uses internally for TRAINING label files

box_xywh = [125, 105, 150, 150]  # center=(125,105), width=150, height=150

# Drawing a box on an image using the xyxy format (most common for display)

image = cv2.imread('bus.jpg')
x1, y1, x2, y2 = box_xyxy

cv2.rectangle(
    image, 
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.putText(image, "Bus: 0.94", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def xywh_to_xyxy(x_center, y_center, w, h):
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]