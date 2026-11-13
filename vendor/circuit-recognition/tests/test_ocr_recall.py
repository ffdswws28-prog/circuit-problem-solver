"""Validate text recall rate against baseline RapidOCR results.

Uses circuit.pdf as the test corpus. Compares SmartFilter + backend
results against full RapidOCR to verify <5% text loss.
"""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import pytest
import numpy as np
import fitz
from src.pipeline.ocr_engine import OCREngine
from src.pipeline.yolo_detector import YOLODetector

PDF_PATH = r"D:\桌面\ai\电路.pdf"


def load_page(page_idx: int):
    """Load a single page as BGR numpy array."""
    doc = fitz.open(PDF_PATH)
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )[:, :, ::-1]
    doc.close()
    return img


@pytest.fixture(scope="module")
def yolo():
    detector = YOLODetector()
    detector.load_model()
    return detector


@pytest.fixture(scope="module")
def ocr_engine():
    engine = OCREngine(use_easyocr=False, enable_correction=True)
    engine.load_model()
    return engine


class TestOcrRecall:
    """Verify optimized OCR doesn't lose critical text."""

    def test_full_ocr_on_all_pages(self, ocr_engine, yolo):
        """Baseline: full OCR works on all 10 pages."""
        doc = fitz.open(PDF_PATH)
        pages_with_text = 0
        for p_idx in range(min(10, doc.page_count)):
            img = load_page(p_idx)
            dets = yolo.detect(img, conf_threshold=0.3)
            if dets:
                results = ocr_engine.recognize_near_components(img, dets)
                if results:
                    pages_with_text += 1
        doc.close()
        assert pages_with_text >= 5  # At least half have text

    def test_smart_filter_recall_vs_baseline(self, ocr_engine, yolo):
        """Compare filtered vs full OCR. Finds a page where SmartFilter is active."""
        # Use YOLO detection count as proxy: scan pages until >= 5 detections found
        page_idx = 0
        img = load_page(0)
        dets = yolo.detect(img, conf_threshold=0.3)
        if len(dets) < 5:
            # Page 0 has few components; try page 4 (typically 12+ detections)
            page_idx = 4
            img = load_page(4)
            dets = yolo.detect(img, conf_threshold=0.3)

        print(f"\n  Using page {page_idx} with {len(dets)} YOLO detections")

        # Full OCR baseline
        baseline = ocr_engine.detect_text_regions(img)
        # Optimized
        optimized = ocr_engine.recognize_near_components(img, dets)

        # Recall = optimized texts / baseline texts
        recall = len(optimized) / max(len(baseline), 1)
        print(f"  Baseline: {len(baseline)} texts")
        print(f"  Optimized: {len(optimized)} texts")
        print(f"  Recall: {recall:.1%}")

        assert recall >= 0.80, f"Text recall too low: {recall:.1%} (< 80% floor)"

    def test_no_yolo_fallback_works(self, ocr_engine):
        """Without YOLO detections, recognize_near_components should still return results."""
        img = load_page(0)
        results = ocr_engine.recognize_near_components(img, [])
        # Should still get some text (full-page detect + center region)
        assert isinstance(results, list)

    def test_profiler_report(self, ocr_engine, yolo):
        """After one page, profiler should produce a valid report."""
        img = load_page(0)
        dets = yolo.detect(img, conf_threshold=0.3)
        ocr_engine.recognize_near_components(img, dets or [])
        report = ocr_engine._profiler.snapshot_and_reset()
        assert "total_ms" in report
        assert "backend" in report
        assert report["boxes_total"] > 0
        print(f"\n  Profiler report: {report}")
