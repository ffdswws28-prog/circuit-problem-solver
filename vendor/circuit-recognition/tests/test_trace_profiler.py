"""Test TraceProfiler — structured timing and loss tracking"""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import time
import pytest
from src.pipeline.trace_profiler import TraceProfiler


class TestTraceProfilerBasic:
    """Core timing and context manager"""

    def test_context_manager_times_block(self):
        profiler = TraceProfiler("test_backend")
        with profiler.trace("det"):
            time.sleep(0.02)
        snap = profiler.snapshot_and_reset()
        assert snap["backend"] == "test_backend"
        assert snap["det_ms"] > 0
        assert snap["det_ms"] < 200  # shouldn't take 200ms

    def test_manual_start_end(self):
        profiler = TraceProfiler("test_backend")
        profiler.start("filter")
        time.sleep(0.01)
        profiler.end("filter")
        snap = profiler.snapshot_and_reset()
        assert snap["filter_ms"] > 0

    def test_nested_contexts(self):
        profiler = TraceProfiler("test_backend")
        with profiler.trace("outer"):
            time.sleep(0.01)
            with profiler.trace("inner"):
                time.sleep(0.01)
        snap = profiler.snapshot_and_reset()
        assert snap["total_ms"] > snap.get("outer_ms", 0)


class TestTraceProfilerCounters:
    """Counters and loss tracking"""

    def test_record_filter_loss_computes_ratio(self):
        profiler = TraceProfiler("test_backend")
        profiler.record_filter_loss(20, 8)
        snap = profiler.snapshot_and_reset()
        assert snap["boxes_total"] == 20
        assert snap["boxes_passed"] == 8
        assert snap["boxes_filtered"] == 12
        assert snap["loss_ratio"] == pytest.approx(0.6, abs=0.001)

    def test_record_fallback_increments(self):
        profiler = TraceProfiler("test_backend")
        profiler.record_fallback()
        profiler.record_fallback(3)
        snap = profiler.snapshot_and_reset()
        assert snap["fallback_count"] == 4

    def test_snapshot_reset_clears(self):
        profiler = TraceProfiler("test_backend")
        profiler.record_filter_loss(10, 5)
        snap1 = profiler.snapshot_and_reset()
        assert snap1["boxes_total"] == 10
        snap2 = profiler.snapshot_and_reset()
        assert snap2["boxes_total"] == 0
