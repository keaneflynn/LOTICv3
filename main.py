import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO('media/yolov8n.pt')
result = model(0, agnostic_nms=True)[0]
detections = sv.Detections.from_yolov8(result)
