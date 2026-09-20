import ultralytics
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model("https://ultralytics.com/images/bus.jpg")

# print(results)

for box in results[0].boxes:
    print(f"Class: {model.names[int(box.cls)]}",
          f"Confidence: {float(box.conf)}",
          f"Box Coords: {box.xyxy}")
    