import cv2

def calculate_IoU(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection 
    
    return intersection/union

image = cv2.imread("bus.jpg")

person_88 = [47, 399, 239, 904]   # cream jacket person
person_86 = [223, 408, 344, 860]  # black jacket person

x1, y1, x2, y2 = person_88

cv2.rectangle(
    image, 
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

x1, y1, x2, y2 = person_86

cv2.rectangle(
    image, 
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.imshow("Detection: ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

iou = calculate_IoU(person_88, person_86)
print(f"IoU between the two people's boxes: {iou:.3f}")

# These two people are standing close together but are separate objects -
# expect a LOW IoU (little to no meaningful overlap)

# Compare: IoU of a box against itself is always 1.0

print(calculate_IoU(person_88, person_88))  # 1.0