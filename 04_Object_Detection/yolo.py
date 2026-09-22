from ultralytics import YOLO

# Load a pretrained YOLO model (trained on COCO - 80 common object classes)
model = YOLO("yolo11n.pt")  # 'n' = nano, smallest/fastest variant

# Run on a single image
results = model("bus.jpg")

# Run on a video file
results = model("my_video.mp4", save=True)

# Run on a live webcam feed (source=0 is usually your default camera)
results = model(source=0, show=True)  # opens a live window with detections drawn

# Train YOLO on YOUR OWN custom dataset (covered in the project below)
model.train(data="my_dataset.yaml", epochs=50, imgsz=640)