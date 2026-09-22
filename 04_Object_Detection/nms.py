import cv2

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    
    intersection = max(0, x2 - x1)  * max(0, y2 - y1)
    
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    union = area1 + area2 - intersection
    
    if union > 0:
        return intersection / union 
    else:
        0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # boxes: list of [x1,y1,x2,y2], scores: matching confidence values
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices.pop(0)   # take the highest-confidence remaining box
        keep.append(current)
        # Remove any remaining box that overlaps too much with this one
        indices = [i for i in indices 
                   if calculate_iou(boxes[current], boxes[i]) < iou_threshold]

    return keep


# Simulated example: 3 raw boxes all detecting the SAME bus, before cleanup
raw_boxes  = [[10, 10, 100, 100], [15, 12, 105, 98], [200, 200, 250, 250]]
raw_scores = [0.95, 0.88, 0.76]   # confidences

kept_indices = non_max_suppression(raw_boxes, raw_scores, iou_threshold=0.5)
print("Boxes kept after NMS:", kept_indices)