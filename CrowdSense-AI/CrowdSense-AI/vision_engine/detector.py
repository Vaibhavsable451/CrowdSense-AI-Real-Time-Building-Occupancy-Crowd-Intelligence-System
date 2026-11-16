"""Phase 2 - YOLO11 person detection."""
from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class 0 = person


class PersonDetector:
    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 640, device: str = "cpu"):
        self.model = YOLO(weights)  # auto-downloads on first run
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame, classes=[PERSON_CLASS_ID], conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


class DetectionAnnotator:
    def __init__(self):
        self.box = sv.BoxAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = [f"person {c:.2f}" for c in detections.confidence]
        out = self.box.annotate(frame.copy(), detections)
        return self.label.annotate(out, detections, labels=labels)
