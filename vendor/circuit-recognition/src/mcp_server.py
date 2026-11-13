"""MCP Server - Circuit Recognition System for Claude Code

Architecture:
  Stage 0: Preprocessor (image load + resize)
  Stage 1: YOLO component detection (TensorRT/PT)
  Stage 2: SAM fine segmentation (on-demand, skip rectangular)
  Stage 3: OCR text recognition (EasyOCR + circuit corrector)
  Stage 4: Topology analysis (coarse parallel + fine fusion)
  Stage 5: Fusion (spatial IOU, label/value matching, netlist)
"""
import asyncio
import json
import os
import sys
import time
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.errors import ErrorCode, CircuitRecognitionError


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline components (model resident, shared across calls)
# ---------------------------------------------------------------------------
_yolo = None
_ocr = None
_sam = None
_cache = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from src.pipeline.yolo_detector import YOLODetector
        _yolo = YOLODetector()
        _yolo.load_model()
    return _yolo


def get_ocr():
    global _ocr
    if _ocr is None:
        from src.pipeline.ocr_engine import OCREngine
        _ocr = OCREngine(use_easyocr=False)
        _ocr.load_model()
    return _ocr


def get_sam():
    global _sam
    if _sam is None:
        from src.pipeline.sam_segmentor import SAMSegmentor
        _sam = SAMSegmentor()
        _sam.load_model()
    return _sam


def get_cache():
    global _cache
    if _cache is None:
        from src.cache import MultiLevelCache
        _cache_dir = Path(__file__).parent.parent / "cache"
        _cache = MultiLevelCache(disk_cache_dir=str(_cache_dir))
    return _cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_hash(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_error(code: ErrorCode, detail: str = None) -> dict:
    return {
        "success": False,
        "error": code.code,
        "message": f"[{code.code}] {code.message}" + (f": {detail}" if detail else ""),
    }


def validate_image_path(image_path: str) -> Path:
    """Validate and resolve image path. Raises CircuitRecognitionError on failure."""
    if not image_path:
        raise CircuitRecognitionError(ErrorCode.PARAM_ERROR, "image_path is required")
    path = Path(image_path)
    if not path.exists():
        raise CircuitRecognitionError(ErrorCode.FILE_NOT_FOUND, str(image_path))
    suffix = path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".pdf"):
        raise CircuitRecognitionError(ErrorCode.FILE_FORMAT_NOT_SUPPORTED, suffix)
    return path


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
server = Server("circuit-recognition")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="analyze_circuit",
            description=(
                "Complete circuit diagram analysis: detect components, segment shapes, "
                "recognize text (labels/values), analyze wire topology, and output "
                "structured JSON with netlist."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to circuit image (JPG/PNG/BMP) or PDF",
                    },
                    "enable_sam": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable SAM fine segmentation for irregular components",
                    },
                    "conf_threshold": {
                        "type": "number",
                        "default": 0.5,
                        "description": "YOLO detection confidence threshold (lower = more detections)",
                    },
                    "enable_topology": {
                        "type": "boolean",
                        "default": True,
                        "description": "Analyze wire connections and extract netlist",
                    },
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="detect_components",
            description="Fast component detection only (YOLO bounding boxes + class names)",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "conf_threshold": {"type": "number", "default": 0.4},
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="recognize_text",
            description=(
                "OCR text recognition on circuit diagram. Returns all text regions "
                "with positions and confidence scores. Circuit-specific correction "
                "applied (O->0, I->1, u->mu)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="get_cache_status",
            description="Check if circuit analysis for a file is already cached",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_hash": {
                        "type": "string",
                        "description": "SHA256 hash of the image file content",
                    },
                },
                "required": ["file_hash"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "analyze_circuit":
            return await handle_analyze(arguments)
        elif name == "detect_components":
            return await handle_detect(arguments)
        elif name == "recognize_text":
            return await handle_ocr(arguments)
        elif name == "get_cache_status":
            return await handle_cache(arguments)
        else:
            return [TextContent(
                type="text",
                text=json.dumps(make_error(ErrorCode.PARAM_ERROR, f"Unknown tool: {name}"), ensure_ascii=False),
            )]
    except CircuitRecognitionError as e:
        return [TextContent(type="text", text=json.dumps(make_error(e.error_code, e.detail), ensure_ascii=False))]
    except Exception as e:
        tb = traceback.format_exc()[:500]
        print(f"[MCP] Unhandled error: {tb}", file=sys.stderr)
        return [TextContent(type="text", text=json.dumps(make_error(ErrorCode.UNKNOWN_ERROR, str(e)[:300]), ensure_ascii=False))]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
async def handle_analyze(args: dict):
    """Full pipeline: YOLO -> SAM -> OCR -> Topology -> Fusion."""
    t_start = time.time()

    # --- Validate input ---
    image_path = validate_image_path(args.get("image_path", ""))

    # --- Check cache ---
    file_hash = compute_hash(str(image_path))
    cache = get_cache()
    cached = cache.get(file_hash)
    if cached:
        cached["from_cache"] = True
        return [TextContent(type="text", text=json.dumps(cached, ensure_ascii=False, indent=2))]

    # --- Load image ---
    from src.pipeline.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    img = preprocessor.load_image(str(image_path))

    conf_threshold = args.get("conf_threshold", 0.5)
    enable_sam = args.get("enable_sam", True)
    enable_topology = args.get("enable_topology", True)
    warnings: List[str] = []

    # --- Stage 1: YOLO detection ---
    t1 = time.time()
    try:
        yolo = get_yolo()
        detections = yolo.detect(img, conf_threshold=conf_threshold)
    except Exception as e:
        raise CircuitRecognitionError(ErrorCode.YOLO_DETECTION_FAILED, str(e)[:200])

    yolo_time = time.time() - t1

    # Low-confidence fallback if no detections
    low_conf_detections: List[Dict] = []
    if not detections:
        try:
            low_conf_detections = yolo.detect_low_conf(img)
            if low_conf_detections:
                warnings.append(f"LOW_CONF_FALLBACK: {len(low_conf_detections)} low-confidence detections used")
        except Exception:
            pass

    # --- Stage 2: SAM segmentation (on-demand) ---
    sam_results: List[Dict] = []
    sam_time = 0.0
    if enable_sam and detections:
        t_sam = time.time()
        try:
            sam = get_sam()
            seg_targets = [d for d in detections if not sam.should_skip(d.get("class", ""))]
            if seg_targets:
                sam_results = sam.segment(img, seg_targets)
        except Exception as e:
            warnings.append(f"SAM_FALLBACK: segmentation failed ({e}), using bounding boxes")
        sam_time = time.time() - t_sam

    # --- Stage 3: OCR (optimized — det on full image, rec only near components) ---
    t3 = time.time()
    try:
        ocr = get_ocr()
        all_dets = detections + low_conf_detections

        # Use optimized OCR: detect text boxes on full image,
        # but only RECOGNIZE text near YOLO components (2-5x faster)
        ocr_results = ocr.recognize_near_components(img, all_dets)
    except Exception as e:
        raise CircuitRecognitionError(ErrorCode.OCR_RECOGNIZE_FAILED, str(e)[:200])
    ocr_time = time.time() - t3

    # --- Stage 4: Topology ---
    topo_results: Dict = {}
    topo_time = 0.0
    if enable_topology:
        t4 = time.time()
        from src.pipeline.topology import TopologyAnalyzer
        topo = TopologyAnalyzer()
        coarse = topo.coarse_analysis(img)
        # Build component list with OCR labels for fine analysis
        from src.pipeline.fusion import FusionEngine
        temp_fusion = FusionEngine()
        comps_with_labels = temp_fusion._associate_ocr(
            [{"bbox": d["bbox"], "class": d["class"], "confidence": d["confidence"]} for d in detections],
            ocr_results,
            [],  # no warnings accumulator here
        )
        topo_results = topo.fine_analysis(comps_with_labels, ocr_results, coarse)
        topo_time = time.time() - t4

    # --- Stage 5: Fusion ---
    t5 = time.time()
    from src.pipeline.fusion import FusionEngine
    fusion = FusionEngine()
    result = fusion.fuse(
        yolo_results=detections,
        sam_results=sam_results,
        ocr_results=ocr_results,
        topo_results=topo_results,
        low_conf_detections=low_conf_detections,
    )

    # Merge our warnings
    result["warnings"] = warnings + result.get("warnings", [])
    fusion_time = time.time() - t5

    total_time = time.time() - t_start

    # --- Build final output ---
    output = {
        "success": True,
        "file_hash": file_hash,
        "image_path": str(image_path),
        "components": result.get("components", []),
        "connections": result.get("connections", []),
        "nets": result.get("nets", {}),
        "title_block": result.get("title_block", {}),
        "summary": result.get("summary", ""),
        "warnings": result.get("warnings", []),
        "timing": {
            "yolo": round(yolo_time, 3),
            "sam": round(sam_time, 3),
            "ocr": round(ocr_time, 3),
            "topology": round(topo_time, 3),
            "fusion": round(fusion_time, 3),
            "total": round(total_time, 3),
        },
    }

    # --- Cache result ---
    try:
        cache.set(file_hash, output)
    except Exception:
        warnings.append("CACHE_WRITE_FAILED: result not cached")

    return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]


async def handle_detect(args: dict):
    """Component detection only."""
    image_path = validate_image_path(args.get("image_path", ""))

    from src.pipeline.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    img = preprocessor.load_image(str(image_path))

    try:
        yolo = get_yolo()
        detections = yolo.detect(img, conf_threshold=args.get("conf_threshold", 0.5))
    except Exception as e:
        raise CircuitRecognitionError(ErrorCode.YOLO_DETECTION_FAILED, str(e)[:200])

    # Add fallback
    if not detections:
        try:
            detections = yolo.detect_low_conf(img)
        except Exception:
            pass

    return [TextContent(type="text", text=json.dumps({
        "success": True,
        "component_count": len(detections),
        "components": detections,
    }, ensure_ascii=False, indent=2))]


async def handle_ocr(args: dict):
    """Text recognition only."""
    image_path = validate_image_path(args.get("image_path", ""))

    from src.pipeline.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    img = preprocessor.load_image(str(image_path))

    try:
        ocr = get_ocr()
        texts = ocr.recognize_all(img)
    except Exception as e:
        raise CircuitRecognitionError(ErrorCode.OCR_RECOGNIZE_FAILED, str(e)[:200])

    return [TextContent(type="text", text=json.dumps({
        "success": True,
        "text_count": len(texts),
        "texts": texts,
    }, ensure_ascii=False, indent=2))]


async def handle_cache(args: dict):
    """Cache status check."""
    file_hash = args.get("file_hash", "")
    if not file_hash:
        return [TextContent(type="text", text=json.dumps(make_error(ErrorCode.PARAM_ERROR, "file_hash is required"), ensure_ascii=False))]

    cache = get_cache()
    hit = cache.get(file_hash) is not None
    return [TextContent(type="text", text=json.dumps({
        "success": True,
        "cached": hit,
        "file_hash": file_hash,
    }, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    print("[MCP] Circuit Recognition Server starting...", file=sys.stderr)

    # System info
    try:
        import torch
        print(f"[MCP] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
        if torch.cuda.is_available():
            print(f"[MCP] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB", file=sys.stderr)
    except Exception as e:
        print(f"[MCP] PyTorch check: {e}", file=sys.stderr)

    # Pre-warm YOLO (most critical path)
    print("[MCP] Pre-warming YOLO...", file=sys.stderr)
    try:
        get_yolo()
        print("[MCP] YOLO ready", file=sys.stderr)
    except Exception as e:
        print(f"[MCP] YOLO warmup failed (will retry on first call): {e}", file=sys.stderr)

    print("[MCP] Server ready (stdio protocol)", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
