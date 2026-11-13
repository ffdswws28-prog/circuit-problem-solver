"""Stage 5: Result fusion -- spatial association, label matching, topology, output standardization"""
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Component class mapping (COCO pre-trained names -> circuit domain names)
# ---------------------------------------------------------------------------
CLASS_ALIASES = {
    "resistor": "resistor",
    "capacitor": "capacitor",
    "inductor": "inductor",
    "diode": "diode",
    "transistor": "transistor",
    "mosfet": "mosfet",
    "opamp": "opamp",
    "transformer": "transformer",
    "connector": "connector",
    "ic": "ic",
    "cell phone": "ic",
    "tv": "ic",
    "laptop": "ic",
    "keyboard": "ic",
    "sports ball": "circle",
    "bottle": "capacitor",
}

# Component label patterns: prefix + digits
_LABEL_RE = re.compile(
    r'^(?P<prefix>[RrCcLlDdQqUuJjPpTtXxYyKkFfMm])(?P<num>\d+[A-Za-z]?)$'
)

# Value patterns: number + unit
_VALUE_RE = re.compile(
    r'^(?P<value>\d+\.?\d*)\s*'
    r'(?P<unit>[kKMm]?[ΩΩ]?|[μuUnNpPmM]?[FfHh]|[kKMmGg]?[VvAaWw]|'
    r'[kKMmGg]?[ΩΩ]|[Hh][Zz]|'
    r'[Oo][Hh][Mm]?)$'
)

# IC part number patterns
_PARTNO_RE = re.compile(
    r'^(?:[1I][nN]\d+|[Ll][Mm]\d+|[Nn][Ee]\d+|[Tt][Ll]\d+|'
    r'[Uu][Aa]\d+|[Mm][Aa][Xx]\d+|[Aa][Dd]\d+|[Ll][Tt]\d+|'
    r'[Oo][Pp]\d+|[2][Nn]\d+|[7][4][Ll][SsVvHhCc]\d+|'
    r'[4][0]\d+|[Cc][Dd]4\d+)$'
)

# Net label patterns: VCC, GND, VDD, VSS, VIN, VOUT, etc.
_NET_RE = re.compile(
    r'^(?:VCC|VDD|VSS|VEE|GND|VIN|VOUT|VREF|VBAT|V[\w]*|'
    r'NET\w*|CLK|RST|EN|PWM|TX|RX|SDA|SCL|MISO|MOSI)$',
    re.IGNORECASE,
)


class FusionEngine:
    """Fuses YOLO, SAM, OCR, and topology results into structured circuit output."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def compute_hash(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def fuse(
        self,
        yolo_results: List[Dict],
        sam_results: List[Dict],
        ocr_results: List[Dict],
        topo_results: Dict,
        title_block: Optional[Dict] = None,
        low_conf_detections: Optional[List[Dict]] = None,
    ) -> Dict:
        """Fuse all pipeline outputs into structured JSON + human-readable summary.

        Parameters
        ----------
        yolo_results : list of {bbox, class, confidence}
        sam_results  : list of {contour, score, fallback?}
        ocr_results  : list of {bbox, text, confidence, original?}
        topo_results : dict with keys 'coarse' and 'fine'
        title_block  : optional dict {text, bbox}
        low_conf_detections : optional fallback detections for gap-filling

        Returns
        -------
        dict with keys: components, connections, nets, title_block, summary, warnings
        """
        warnings: List[str] = []

        # ----- 1. Deduplicate + merge fallback detections -----
        components = self._dedup_detections(yolo_results, low_conf_detections or [])

        if not components:
            warnings.append("NO_VALID_COMPONENTS: 0 components detected")

        # ----- 2. Associate OCR text with components -----
        components = self._associate_ocr(components, ocr_results, warnings)

        # ----- 3. Merge SAM contours -----
        if sam_results:
            self._merge_sam_contours(components, sam_results)

        # ----- 4. Build connections from topology -----
        fine = topo_results.get("fine", topo_results)
        connections, nets = self._build_connections(components, fine)

        # ----- 5. Extract unassociated net labels -----
        net_labels = self._find_net_labels(ocr_results, components)
        nets = self._merge_nets(nets, net_labels)

        # ----- 6. Generate summary -----
        summary = self._generate_summary(components, connections, nets)

        return {
            "components": components,
            "connections": connections,
            "nets": nets,
            "title_block": title_block or {},
            "summary": summary,
            "warnings": warnings if warnings else [],
        }

    # ------------------------------------------------------------------
    # Detection dedup + merge
    # ------------------------------------------------------------------
    @staticmethod
    def _dedup_detections(
        primary: List[Dict], fallback: List[Dict], iou_thresh: float = 0.5
    ) -> List[Dict]:
        """Merge primary and fallback detections, removing duplicates via IOU."""
        merged = list(primary)
        for fb in fallback:
            fb_box = fb["bbox"]
            duplicate = False
            for m in merged:
                if FusionEngine._iou(fb_box, m["bbox"]) > iou_thresh:
                    # Keep the higher-confidence one
                    if fb["confidence"] > m["confidence"]:
                        m.update(fb)
                    duplicate = True
                    break
            if not duplicate:
                merged.append(fb)
        return merged

    # ------------------------------------------------------------------
    # OCR spatial association
    # ------------------------------------------------------------------
    def _associate_ocr(
        self, components: List[Dict], ocr_items: List[Dict], warnings: List[str]
    ) -> List[Dict]:
        """For each component, find nearby OCR text and classify as label / value / pin."""
        # Group OCR items that haven't been matched yet
        unmatched = list(ocr_items)

        for comp in components:
            bx, by, bw, bh = comp["bbox"]
            # Expand search region around component
            search_bbox = [
                int(bx - bw * 0.4),
                int(by - bh * 1.2),  # text is usually above or beside
                int(bw * 1.8),
                int(bh * 2.4),
            ]

            candidates = []
            for ocr in unmatched[:]:
                if ocr.get("bbox") is None:
                    continue
                ox, oy, ow, oh = ocr["bbox"]
                ocr_cx, ocr_cy = ox + ow / 2, oy + oh / 2
                if self._point_in_bbox(ocr_cx, ocr_cy, search_bbox):
                    # Compute distance score (closer = better)
                    comp_cx, comp_cy = bx + bw / 2, by + bh / 2
                    dist = ((ocr_cx - comp_cx) ** 2 + (ocr_cy - comp_cy) ** 2) ** 0.5
                    candidates.append((dist, ocr))

            # Sort by distance
            candidates.sort(key=lambda x: x[0])

            label = None
            value = None
            pins: List[str] = []

            for _, ocr in candidates:
                text = ocr.get("text", "").strip()
                if not text:
                    continue

                # Classify the text
                if label is None and self._is_label(text):
                    label = text
                    unmatched = [x for x in unmatched if x is not ocr]
                elif value is None and self._is_value(text):
                    value = text
                    unmatched = [x for x in unmatched if x is not ocr]
                elif self._is_pin_number(text) and text not in pins:
                    pins.append(text)
                    unmatched = [x for x in unmatched if x is not ocr]

            comp["label"] = label
            comp["value"] = value
            comp["pins"] = pins

            # Annotate confidence
            if label is None:
                comp.setdefault("warnings", []).append("no_label")

        # Any remaining unmatched OCR text -> possible net labels or orphan text
        return components

    # ------------------------------------------------------------------
    # SAM contour merge
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_sam_contours(components: List[Dict], sam_results: List[Dict]):
        """Attach SAM contours to components (by index alignment)."""
        comps_with_contour = [c for c in components if c.get("class", "").lower() not in SAMSegmentor.SKIP_CLASSES]
        for i, comp in enumerate(comps_with_contour):
            if i < len(sam_results):
                comp["contour"] = sam_results[i].get("contour")
                comp["contour_score"] = sam_results[i].get("score")
                if sam_results[i].get("fallback"):
                    comp.setdefault("warnings", []).append("sam_fallback")

    # ------------------------------------------------------------------
    # Connections & nets
    # ------------------------------------------------------------------
    @staticmethod
    def _build_connections(components: List[Dict], fine_topo: Dict) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """Build connection list and net dictionary."""
        # If topology analyzer provided connections, use them
        raw_connections = fine_topo.get("connections", []) if fine_topo else []
        nets: Dict[str, List[str]] = {}

        connections: List[Dict] = []
        for conn in raw_connections:
            from_ref = conn.get("from", "")
            to_ref = conn.get("to", "")
            net_name = conn.get("net", "")

            connections.append({
                "from": from_ref,
                "to": to_ref,
                "net": net_name or None,
                "type": conn.get("type", "wire"),
            })

            if net_name:
                nets.setdefault(net_name, [])
                for ref in (from_ref, to_ref):
                    if ref and ref not in nets[net_name]:
                        nets[net_name].append(ref)

        # If no topology connections, build simple proximity-based connections
        if not connections and len(components) >= 2:
            connections = FusionEngine._proximity_connections(components)

        return connections, nets

    @staticmethod
    def _proximity_connections(components: List[Dict], max_gap: int = 100) -> List[Dict]:
        """Fallback: connect components that are close to each other (horizontal/vertical)."""
        connections: List[Dict] = []
        for i, a in enumerate(components):
            for j, b in enumerate(components):
                if j <= i:
                    continue
                ax, ay, aw, ah = a["bbox"]
                bx, by, bw, bh = b["bbox"]

                # Check horizontal alignment (same Y range, close X)
                a_cx, a_cy = ax + aw / 2, ay + ah / 2
                b_cx, b_cy = bx + bw / 2, by + bh / 2

                h_gap = abs(a_cx - b_cx) - (aw + bw) / 2
                v_gap = abs(a_cy - b_cy) - (ah + bh) / 2

                if h_gap < max_gap and abs(a_cy - b_cy) < (ah + bh) / 2 + 20:
                    a_ref = a.get("label") or f"comp_{i}"
                    b_ref = b.get("label") or f"comp_{j}"
                    connections.append({
                        "from": a_ref,
                        "to": b_ref,
                        "net": None,
                        "type": "proximity",
                    })
                elif v_gap < max_gap and abs(a_cx - b_cx) < (aw + bw) / 2 + 20:
                    a_ref = a.get("label") or f"comp_{i}"
                    b_ref = b.get("label") or f"comp_{j}"
                    connections.append({
                        "from": a_ref,
                        "to": b_ref,
                        "net": None,
                        "type": "proximity",
                    })
        return connections

    # ------------------------------------------------------------------
    # Net labels
    # ------------------------------------------------------------------
    @staticmethod
    def _find_net_labels(ocr_items: List[Dict], components: List[Dict]) -> Dict[str, List[str]]:
        """Find unassociated net labels (VCC, GND, etc.) in OCR results."""
        net_labels: Dict[str, List[str]] = {}
        associated_texts = set()
        for c in components:
            if c.get("label"):
                associated_texts.add(c["label"])
            if c.get("value"):
                associated_texts.add(c["value"])

        for ocr in ocr_items:
            text = ocr.get("text", "").strip()
            if text in associated_texts:
                continue
            if _NET_RE.match(text):
                net_labels.setdefault(text.upper(), []).append(text)

        return net_labels

    @staticmethod
    def _merge_nets(
        existing: Dict[str, List[str]], net_labels: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Merge topology nets with OCR net labels."""
        merged = dict(existing)
        for name, labels in net_labels.items():
            merged.setdefault(name, []).extend(labels)
        return merged

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_summary(
        components: List[Dict], connections: List[Dict], nets: Dict[str, List[str]]
    ) -> str:
        parts = [f"Total components: {len(components)}"]

        type_counts: Dict[str, int] = {}
        for c in components:
            cls = c.get("class", "unknown")
            cls = CLASS_ALIASES.get(cls, cls)
            type_counts[cls] = type_counts.get(cls, 0) + 1

        for cls, count in sorted(type_counts.items()):
            parts.append(f"{cls}: {count}")

        parts.append(f"Connections: {len(connections)}")
        if nets:
            parts.append(f"Nets: {', '.join(sorted(nets.keys()))}")
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Text classification helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_label(text: str) -> bool:
        """Check if text looks like a component label (R1, C2, U3A, etc.)."""
        return bool(_LABEL_RE.match(text))

    @staticmethod
    def _is_value(text: str) -> bool:
        """Check if text looks like a component value (10k, 100uF, 1N4148, etc.)."""
        if _VALUE_RE.match(text):
            return True
        if _PARTNO_RE.match(text):
            return True
        # Pure numeric value
        if re.match(r'^\d+\.?\d*$', text):
            return True
        return False

    @staticmethod
    def _is_pin_number(text: str) -> bool:
        """Check if text is a pin number (1-64, sometimes with letter suffix)."""
        return bool(re.match(r'^\d{1,2}[A-Za-z]?$', text))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        """Intersection over Union for [x, y, w, h] boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix = max(0, min(ax2, bx2) - max(ax, bx))
        iy = max(0, min(ay2, by2) - max(ay, by))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area_a = aw * ah
        area_b = bw * bh
        return inter / (area_a + area_b - inter + 1e-8)

    @staticmethod
    def _point_in_bbox(px: float, py: float, bbox: List[int]) -> bool:
        x, y, w, h = bbox
        return x <= px <= x + w and y <= py <= y + h


# Late import to avoid circular dependency
from src.pipeline.sam_segmentor import SAMSegmentor  # noqa: E402
