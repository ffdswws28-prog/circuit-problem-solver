"""Stage 2: MobileSAM/EdgeSAM on-demand fine segmentation"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict


class SAMSegmentor:
    """SAM segmentor - triggered only for irregular ICs/connectors"""

    SKIP_CLASSES = {"resistor", "capacitor", "diode", "inductor"}

    def __init__(self, model_path: str = None, model_type: str = "vit_b"):
        default_path = Path(__file__).parent.parent.parent / "models" / "sam_vit_b_01ec64.pth"
        self.model_path = model_path or str(default_path)
        self.model_type = model_type
        self.model = None

    def load_model(self):
        """Load SAM model (auto-download on first run, ~358 MB)."""
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        model_file = Path(self.model_path)
        if not model_file.exists():
            print(f"[SAM] Model not found, downloading to {model_file} (~358 MB)...")
            print(f"[SAM] This is a one-time download. Please wait...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            try:
                model_file.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(url, str(model_file))
                print(f"[SAM] Download complete: {model_file}")
            except Exception as e:
                raise RuntimeError(
                    f"SAM model download failed: {e}\n"
                    f"Please manually download from {url} and place at {model_file}"
                )

        self.sam = sam_model_registry[self.model_type](checkpoint=str(model_file))
        self.sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(self.sam)
        print(f"[SAM] Model loaded: {self.model_type}")

    def should_skip(self, component_class: str) -> bool:
        """Skip SAM for regular rectangular components"""
        return component_class.lower() in self.SKIP_CLASSES

    def segment(self, image: np.ndarray, bbox_list: List[Dict]) -> List[Dict]:
        """Segment components, return contour polygons"""
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(image)
        results = []

        for item in bbox_list:
            box = item.get("bbox", [0, 0, 0, 0])
            x, y, w, h = box
            input_box = np.array([x, y, x + w, y + h])

            masks, scores, _ = self.predictor.predict(
                box=input_box[None, :],
                multimask_output=False,
            )
            if masks is not None and len(masks) > 0:
                mask = masks[0].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0].squeeze().tolist()
                    # Ensure at least 3 points for a polygon
                    if len(contour) >= 3:
                        results.append({"contour": contour, "score": float(scores[0])})
                        continue
            # Fallback: use bounding box as rectangle contour
            results.append({
                "contour": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                "score": 1.0,
                "fallback": True,
            })
        return results
