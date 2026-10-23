
from ultralytics import YOLO
import cv2


class ObjectDetector:

    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    # Image object detection
    def detect(self, image, confidence=0.25, iou=0.45):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            verbose=False
        )
        return results[0]

    # Real-time video object tracking
    def track(self, frame, confidence=0.25, iou=0.45):
        results = self.model.track(
            source=frame,
            conf=confidence,
            iou=iou,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        return results[0]

    # Extract detection information
    def get_detections(self, result):
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "class": result.names[class_id],
                "confidence": round(confidence_score, 3),
                "track_id": track_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        return detections

    # Draw bounding boxes and labels
    def annotate(self, result):
        return cv2.cvtColor(
            result.plot(),
            cv2.COLOR_BGR2RGB
        )
