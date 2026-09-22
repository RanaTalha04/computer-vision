# You virtually never hand-calculate mAP yourself - it involves computing
# precision/recall at many thresholds and integrating the curve.
# In practice, you use a library. Ultralytics computes this for you automatically:

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Running validation on a labeled dataset automatically computes mAP
# (requires a dataset with ground-truth labels in YOLO format)
metrics = model.val(data="coco8.yaml")  # coco8 is a tiny built-in sample dataset

print("mAP50:", metrics.box.map50)      # mAP at IoU threshold 0.5 only
print("mAP50-95:", metrics.box.map)     # mAP averaged across IoU thresholds 0.5 to 0.95