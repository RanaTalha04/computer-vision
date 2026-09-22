def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Ground truth (human-labeled, "correct") boxes for one image
ground_truth = [[50, 50, 150, 150], [300, 300, 400, 400]]  # 2 real objects

# Model's predicted boxes
predictions = [[52, 48, 148, 152], [500, 500, 550, 550]]   # 2 predicted boxes

iou_threshold = 0.5
true_positives = 0
matched_gt = set()

for pred in predictions:
    for i, gt in enumerate(ground_truth):
        if i not in matched_gt and calculate_iou(pred, gt) >= iou_threshold:
            true_positives += 1
            matched_gt.add(i)
            break

false_positives = len(predictions) - true_positives   # predicted but wrong/unmatched
false_negatives = len(ground_truth) - true_positives   # real objects the model missed

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print(f"Precision: {precision:.2f}")  # 0.50 - one prediction was a false alarm
print(f"Recall: {recall:.2f}")        # 0.50 - one real object was completely missed