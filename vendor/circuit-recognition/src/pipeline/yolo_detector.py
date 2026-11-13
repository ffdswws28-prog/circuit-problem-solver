"""Stage 1: YOLO component detection + classification"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class YOLODetector:
    """YOLO detector for circuit component localization"""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.4):
        # Fallback chain: v3 (10-class, textbook-tuned) > v2 (10-class) > mixed (8-class) > enhanced > finetuned > pretrained
        v3 = Path(__file__).parent.parent.parent / "models" / "circuit_yolov8n_v3.pt"
        v2 = Path(__file__).parent.parent.parent / "models" / "circuit_yolov8n_v2.pt"
        mixed = Path(__file__).parent.parent.parent / "models" / "circuit_yolov8n_mixed" / "weights" / "best.pt"
        enhanced = Path(__file__).parent.parent.parent / "models" / "circuit_yolov8n_enhanced.pt"
        finetuned = Path(__file__).parent.parent.parent / "models" / "circuit_yolov8n_finetuned.pt"
        pretrained = Path(__file__).parent.parent.parent / "models" / "yolov8n.pt"
        default_path = v3 if v3.exists() else (v2 if v2.exists() else (mixed if mixed.exists() else (enhanced if enhanced.exists() else (finetuned if finetuned.exists() else pretrained))))
        self.model_path = model_path or str(default_path)
        self.conf_threshold = conf_threshold
        self.model = None

    def load_model(self):
        """Load YOLO model (supports .pt and .trt engines)"""
        model_path = Path(self.model_path)
        if model_path.suffix == ".trt":
            from src.inference.trt_engine import TRTEngine
            self.model = TRTEngine(str(model_path))
            self.model.load()
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
        print(f"[YOLO] Model loaded: {model_path}")

    def detect(self, image: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        """Detect components, return list of {bbox, class, confidence}"""
        threshold = conf_threshold or self.conf_threshold
        results = self.model(image, conf=threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, w, h]
                    "class": cls_name,
                    "confidence": round(conf, 4),
                })
        return detections

    def detect_low_conf(self, image: np.ndarray) -> List[Dict]:
        """Fallback: lower threshold to catch missed components"""
        return self.detect(image, conf_threshold=self.conf_threshold * 0.7)
