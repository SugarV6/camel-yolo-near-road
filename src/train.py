"""Train YOLOv8 on camel dataset."""

from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on data.yaml with fixed settings
model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    project="runs",
    name="camel",
)
