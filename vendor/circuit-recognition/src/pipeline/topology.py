"""Stage 4: Topology analysis -- wire tracing, pin matching, netlist extraction"""
import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Net label patterns
# ---------------------------------------------------------------------------
_NET_LABEL_RE = re.compile(
    r'^(?:VCC|VDD|VSS|VEE|GND|AGND|DGND|VIN|VOUT|VREF|VBAT|'
    r'V[\w]*|NET[A-Za-z0-9_]*|CLK|RST|EN|PWM|TX|RX|SDA|SCL|MISO|MOSI|'
    r'CS|INT|IRQ|ALE|RD|WR|OE)$',
    re.IGNORECASE,
)

# Component terminal positions (relative to bbox, normalized 0-1)
# For simple two-terminal components
TERMINAL_POSITIONS = {
    "resistor":     {"left": (0.0, 0.5),  "right": (1.0, 0.5)},
    "capacitor":    {"left": (0.0, 0.5),  "right": (1.0, 0.5)},
    "inductor":     {"left": (0.0, 0.5),  "right": (1.0, 0.5)},
    "diode":        {"anode": (0.0, 0.5), "cathode": (1.0, 0.5)},
    "transistor":   {"base": (0.3, 0.0),  "collector": (0.7, 1.0), "emitter": (0.3, 1.0)},
    "mosfet":       {"gate": (0.3, 0.0),  "drain": (0.7, 1.0), "source": (0.3, 1.0)},
    "opamp":        {"in+": (0.0, 0.3),   "in-": (0.0, 0.7), "out": (1.0, 0.5),
                     "v+": (0.5, 0.0),    "v-": (0.5, 1.0)},
    "connector":    {"pin1": (0.0, 0.2),  "pin2": (0.0, 0.4),
                     "pin3": (0.0, 0.6),  "pin4": (0.0, 0.8)},
    "ic":           {},  # ICs: use OCR pin numbers
    "transformer":  {"pri1": (0.0, 0.3),  "pri2": (0.0, 0.7),
                     "sec1": (1.0, 0.3),  "sec2": (1.0, 0.7)},
}


class TopologyAnalyzer:
    """Two-layer topology analysis: coarse (parallel) + fine (fusion-stage)."""

    def __init__(self):
        self._line_cache = None

    # ------------------------------------------------------------------
    # Coarse analysis (runs in parallel with YOLO / OCR)
    # ------------------------------------------------------------------
    def coarse_analysis(self, image: np.ndarray) -> dict:
        """Extract wire lines, junctions, and connected components from raw image.

        Runs in parallel with YOLO/OCR (no component dependency).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection
        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=8,
        )

        lines: List[Dict] = []
        if lines_raw is not None:
            for ln in lines_raw:
                x1, y1, x2, y2 = ln[0]
                lines.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                })

        # Find wire junctions (intersection points)
        junctions = self._find_junctions(lines)

        # Connected component analysis on wire mask
        wire_mask = np.zeros_like(gray)
        for ln in lines:
            cv2.line(wire_mask, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), 255, 2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wire_mask, connectivity=8)

        nodes = []
        for i in range(1, num_labels):  # skip background (label 0)
            x, y, w, h, area = stats[i]
            if area > 20:  # filter noise
                nodes.append({
                    "id": i,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(area),
                    "centroid": [int(x + w / 2), int(y + h / 2)],
                })

        self._line_cache = {
            "lines": lines,
            "junctions": junctions,
            "nodes": nodes,
        }

        return self._line_cache

    # ------------------------------------------------------------------
    # Fine analysis (runs in fusion stage, has component + OCR data)
    # ------------------------------------------------------------------
    def fine_analysis(
        self,
        components: List[Dict],
        texts: List[Dict],
        coarse_result: dict,
    ) -> dict:
        """Build netlist: assign pins, resolve net labels, generate connections.

        Parameters
        ----------
        components : list of {bbox, class, label, value, pins, ...}
        texts : list of OCR {text, bbox, confidence, ...}
        coarse_result : output of coarse_analysis()

        Returns
        -------
        dict with keys: connections, nets, net_labels_map
        """
        lines = coarse_result.get("lines", [])
        junctions = coarse_result.get("junctions", [])
        nodes = coarse_result.get("nodes", [])

        # 1. Assign terminal positions for each component
        for comp in components:
            comp["terminals"] = self._get_terminals(comp)

        # 2. Match wire endpoints to component terminals
        connections: List[Dict] = []
        nets: Dict[str, List[str]] = {}

        if len(components) >= 2:
            connections, nets = self._match_wires_to_terminals(
                components, lines, junctions, nodes
            )

        # 3. Resolve net labels from OCR text
        net_label_map = self._resolve_net_labels(texts, components, junctions)

        # 4. Merge net labels into nets
        for net_name, pin_refs in net_label_map.items():
            nets.setdefault(net_name, []).extend(pin_refs)

        # 5. Assign net names to connections
        for conn in connections:
            matched_net = self._find_net_for_connection(conn, nets)
            if matched_net:
                conn["net"] = matched_net

        return {
            "connections": connections,
            "nets": nets,
            "net_labels": list(net_label_map.keys()),
            "junction_count": len(junctions),
            "wire_count": len(lines),
        }

    # ------------------------------------------------------------------
    # Terminal computation
    # ------------------------------------------------------------------
    @staticmethod
    def _get_terminals(comp: Dict) -> Dict[str, Tuple[int, int]]:
        """Compute absolute terminal positions from component class and bbox."""
        cls = comp.get("class", "unknown").lower()
        bx, by, bw, bh = comp.get("bbox", [0, 0, 100, 100])
        positions = TERMINAL_POSITIONS.get(cls, {"left": (0.0, 0.5), "right": (1.0, 0.5)})

        terminals = {}
        for name, (rx, ry) in positions.items():
            tx = int(bx + rx * bw)
            ty = int(by + ry * bh)
            terminals[name] = (tx, ty)
        return terminals

    # ------------------------------------------------------------------
    # Wire-to-terminal matching
    # ------------------------------------------------------------------
    def _match_wires_to_terminals(
        self,
        components: List[Dict],
        lines: List[Dict],
        junctions: List[Dict],
        nodes: List[Dict],
        max_dist: int = 30,
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """Match wire endpoints to component terminals, build net connections."""
        connections: List[Dict] = []
        nets: Dict[str, List[str]] = {}

        # Collect all terminal points
        all_terminals: List[Tuple[str, Tuple[int, int], Dict]] = []  # (ref, (x,y), comp)
        for i, comp in enumerate(components):
            label = comp.get("label") or f"U{i}"
            for term_name, (tx, ty) in comp.get("terminals", {}).items():
                ref = f"{label}:{term_name}"
                all_terminals.append((ref, (tx, ty), comp))

        # For each line, check if endpoints are near any terminals
        terminal_connections: Dict[str, List[Tuple[int, int]]] = {}  # ref -> connected wire points

        for ln in lines:
            p1 = (ln["x1"], ln["y1"])
            p2 = (ln["x2"], ln["y2"])

            for ref, (tx, ty), comp in all_terminals:
                d1 = ((p1[0] - tx) ** 2 + (p1[1] - ty) ** 2) ** 0.5
                d2 = ((p2[0] - tx) ** 2 + (p2[1] - ty) ** 2) ** 0.5

                if d1 < max_dist:
                    terminal_connections.setdefault(ref, []).append(p2)
                if d2 < max_dist:
                    terminal_connections.setdefault(ref, []).append(p1)

        # If no wire-terminal matches, use proximity-based connections
        if not terminal_connections:
            return self._proximity_connections(components), {}

        # Build connections: terminals that share wire endpoints are on the same net
        net_index = 0
        term_refs = list(terminal_connections.keys())

        for i in range(len(term_refs)):
            for j in range(i + 1, len(term_refs)):
                ref_a = term_refs[i]
                ref_b = term_refs[j]
                points_a = terminal_connections[ref_a]
                points_b = terminal_connections[ref_b]

                if self._share_connection(points_a, points_b, junctions, max_dist=20):
                    net_name = f"net_{net_index}"
                    connections.append({
                        "from": ref_a,
                        "to": ref_b,
                        "net": net_name,
                        "type": "wire",
                    })
                    nets.setdefault(net_name, []).extend([ref_a, ref_b])
                    net_index += 1

        return connections, nets

    @staticmethod
    def _share_connection(
        pts_a: List[Tuple[int, int]],
        pts_b: List[Tuple[int, int]],
        junctions: List[Dict],
        max_dist: int = 20,
    ) -> bool:
        """Check if two sets of wire endpoints share a connection (direct or via junction)."""
        for pa in pts_a:
            for pb in pts_b:
                if ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 < max_dist:
                    return True
            # Check junction proximity
            for jn in junctions:
                jx, jy = jn["x"], jn["y"]
                if ((pa[0] - jx) ** 2 + (pa[1] - jy) ** 2) ** 0.5 < max_dist:
                    for pb in pts_b:
                        if ((pb[0] - jx) ** 2 + (pb[1] - jy) ** 2) ** 0.5 < max_dist:
                            return True
        return False

    @staticmethod
    def _proximity_connections(components: List[Dict], max_gap: int = 120) -> List[Dict]:
        """Fallback: connect components that are horizontally/vertically aligned."""
        connections: List[Dict] = []
        for i, a in enumerate(components):
            for j, b in enumerate(components):
                if j <= i:
                    continue
                ax, ay, aw, ah = a["bbox"]
                bx, by, bw, bh = b["bbox"]
                a_cx, a_cy = ax + aw / 2, ay + ah / 2
                b_cx, b_cy = bx + bw / 2, by + bh / 2

                h_gap = abs(a_cx - b_cx) - (aw + bw) / 2
                v_gap = abs(a_cy - b_cy) - (ah + bh) / 2

                if h_gap < max_gap and abs(a_cy - b_cy) < (ah + bh) / 2 + 30:
                    a_ref = a.get("label") or f"comp_{i}"
                    b_ref = b.get("label") or f"comp_{j}"
                    connections.append({
                        "from": a_ref,
                        "to": b_ref,
                        "net": None,
                        "type": "proximity",
                    })
                elif v_gap < max_gap and abs(a_cx - b_cx) < (aw + bw) / 2 + 30:
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
    # Net label resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_net_labels(
        texts: List[Dict],
        components: List[Dict],
        junctions: List[Dict],
    ) -> Dict[str, List[str]]:
        """Find net labels (VCC, GND, etc.) and associate them with nearby wire junctions."""
        net_label_map: Dict[str, List[str]] = {}
        associated_texts = set()
        for c in components:
            if c.get("label"):
                associated_texts.add(c["label"])
            if c.get("value"):
                associated_texts.add(c.get("value"))

        for ocr in texts:
            text = ocr.get("text", "").strip()
            if text in associated_texts:
                continue
            if not _NET_LABEL_RE.match(text):
                continue

            net_name = text.upper()
            net_label_map.setdefault(net_name, [])

            # Associate with nearest junction or component terminal
            bbox = ocr.get("bbox")
            if bbox:
                ox, oy, ow, oh = bbox
                ocr_center = (ox + ow / 2, oy + oh / 2)

                # Find nearest junction
                nearest_junction = None
                min_dist = 60  # max association distance
                for jn in junctions:
                    jx, jy = jn["x"], jn["y"]
                    dist = ((ocr_center[0] - jx) ** 2 + (ocr_center[1] - jy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_junction = jn

                if nearest_junction:
                    net_label_map[net_name].append(
                        f"J{nearest_junction.get('id', '?')}"
                    )
                else:
                    net_label_map[net_name].append(text)

        return net_label_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_net_for_connection(conn: Dict, nets: Dict[str, List[str]]) -> Optional[str]:
        """Find which net a connection belongs to."""
        from_ref = conn.get("from", "")
        to_ref = conn.get("to", "")
        for net_name, refs in nets.items():
            if from_ref in refs or to_ref in refs:
                return net_name
        return None

    @staticmethod
    def _find_junctions(lines: List[Dict], proximity: int = 10) -> List[Dict]:
        """Find wire junction points where multiple lines meet."""
        if not lines:
            return []

        # Collect all endpoints
        points: List[Tuple[int, int]] = []
        for ln in lines:
            points.append((ln["x1"], ln["y1"]))
            points.append((ln["x2"], ln["y2"]))

        # Cluster nearby endpoints
        junctions: List[Dict] = []
        used = set()

        for i, p1 in enumerate(points):
            if i in used:
                continue
            cluster = [p1]
            used.add(i)
            for j, p2 in enumerate(points):
                if j in used:
                    continue
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if (dx * dx + dy * dy) ** 0.5 < proximity:
                    cluster.append(p2)
                    used.add(j)

            if len(cluster) >= 2:  # at least 2 endpoints meet
                cx = int(sum(p[0] for p in cluster) / len(cluster))
                cy = int(sum(p[1] for p in cluster) / len(cluster))
                junctions.append({"x": cx, "y": cy, "degree": len(cluster)})

        return junctions
