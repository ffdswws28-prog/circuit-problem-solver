"""Structured per-page OCR pipeline timing and loss tracking."""
import time
from contextlib import contextmanager
from typing import Dict


class TraceProfiler:
    """Collects stage timings, box counts, and fallback events per page.

    Designed for per-page reset via snapshot_and_reset().
    """

    def __init__(self, backend_name: str):
        self._backend = backend_name
        self._times: Dict[str, float] = {}          # tag -> seconds
        self._active: Dict[str, float] = {}          # tag -> start_time
        self._counts: Dict[str, int] = {
            "boxes_total": 0,
            "boxes_passed": 0,
            "boxes_filtered": 0,
            "fallback_count": 0,
        }

    # ---- context manager for clean with-blocks ----

    @contextmanager
    def trace(self, tag: str):
        """Context manager: auto start/end around a block.

        Usage:
            with profiler.trace("rec_batch"):
                results = backend.recognize_batch(img, boxes)
        """
        self.start(tag)
        try:
            yield
        finally:
            self.end(tag)

    # ---- manual start/end (for code that spans function calls) ----

    def start(self, tag: str) -> None:
        self._active[tag] = time.perf_counter()

    def end(self, tag: str) -> None:
        if tag not in self._active:
            return
        elapsed = time.perf_counter() - self._active.pop(tag)
        self._times[tag] = self._times.get(tag, 0.0) + elapsed

    # ---- convenience counters ----

    def record_filter_loss(self, total: int, passed: int) -> None:
        """Record box counts and compute loss_ratio automatically."""
        self._counts["boxes_total"] += total
        self._counts["boxes_passed"] += passed
        self._counts["boxes_filtered"] += (total - passed)

    def record_fallback(self, num: int = 1) -> None:
        """Increment fallback counter."""
        self._counts["fallback_count"] += num

    # ---- snapshot ----

    def snapshot_and_reset(self) -> Dict:
        """Return structured report dict and clear accumulated data."""
        total = sum(self._times.values())
        boxes_total = self._counts["boxes_total"]
        boxes_passed = self._counts["boxes_passed"]
        loss_ratio = 0.0
        if boxes_total > 0:
            loss_ratio = round((boxes_total - boxes_passed) / boxes_total, 4)

        rec_box_count = boxes_passed if boxes_passed > 0 else 1
        rec_per_box = self._times.get("rec_batch", 0.0) / rec_box_count * 1000

        report = {
            "backend": self._backend,
            "det_ms": round(self._times.get("det", 0.0) * 1000, 1),
            "filter_ms": round(self._times.get("filter", 0.0) * 1000, 1),
            "rec_batch_ms": round(self._times.get("rec_batch", 0.0) * 1000, 1),
            "rec_per_box_ms": round(rec_per_box, 1),
            "corrector_ms": round(self._times.get("corrector", 0.0) * 1000, 1),
            "total_ms": round(total * 1000, 1),
            "boxes_total": boxes_total,
            "boxes_passed": boxes_passed,
            "boxes_filtered": self._counts["boxes_filtered"],
            "fallback_count": self._counts["fallback_count"],
            "loss_ratio": loss_ratio,
        }

        # Reset
        self._times.clear()
        for k in self._counts:
            self._counts[k] = 0

        return report
