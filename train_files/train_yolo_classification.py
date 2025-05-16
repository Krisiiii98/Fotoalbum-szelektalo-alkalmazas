from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=256,
    batch=32,
    device=0
)