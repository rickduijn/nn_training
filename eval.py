from ultralytics import YOLO

model = YOLO("runs/segment/runs/bell_pepper_segmentation-2/weights/best.pt")


metrics = model.val()

print("Box mAP:", metrics.box.map)
print("Mask mAP:", metrics.seg.map)