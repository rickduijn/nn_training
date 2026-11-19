from ultralytics import YOLO
import cv2
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "runs/segment/runs/bell_pepper_segmentation-2/weights/best.pt"

# Change this to an image you want to test
IMAGE_PATH = "/home/rick-duijnisveld/oak_project/bell_pepper_dataset_horizontal/images/image_000120.jpg"

CONFIDENCE = 0.5
SHOW_RESULTS = True
SAVE_RESULTS = True


# -----------------------------
# Load model
# -----------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print(f"Classes: {model.names}")


# -----------------------------
# Run inference
# -----------------------------

print("Running inference...")

results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    save=SAVE_RESULTS,
    show=SHOW_RESULTS,
    verbose=True
)


# -----------------------------
# Display results
# -----------------------------

for result in results:

    # Number of detected objects
    if result.boxes is not None:
        num_objects = len(result.boxes)

        print(f"\nDetected objects: {num_objects}")

        for i, box in enumerate(result.boxes):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"Object {i + 1}: "
                f"{class_name}, "
                f"confidence={confidence:.2f}, "
                f"bbox={[round(x1), round(y1), round(x2), round(y2)]}"
            )

    # Check segmentation masks
    if result.masks is not None:
        print(f"Segmentation masks: {len(result.masks)}")
    else:
        print("No segmentation masks detected.")

print("\nInference completed!")