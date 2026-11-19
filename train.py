from ultralytics import YOLO

# Load pretrained nano segmentation model
model = YOLO("yolov8n-seg.pt")

# Train on bell pepper dataset
results = model.train(
    data="dataset/data.yaml",
    epochs=150,
    imgsz=512,
    batch=4,
    device="cpu",
    workers=4,
    project="runs",
    name="bell_pepper_segmentation"
)