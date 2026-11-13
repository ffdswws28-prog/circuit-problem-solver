"""Test TrtRecognizer -- batch grouping and padding logic"""
import sys
sys.path.insert(0, r"D:\ai\circuit-recognition")
sys.path.insert(0, r"D:\ai\circuit-recognition\src")

import numpy as np
import pytest


# We test the helper functions directly -- no TRT context needed
class TestTrtRecognizerBatchGrouping:
    """Test crop grouping and batch construction logic"""

    def test_group_by_width_step_64(self):
        from src.pipeline.trt_recognizer import TrtRecognizer
        rec = TrtRecognizer(batch_group_step=64)
        # Create crops of various widths: 50, 100, 200, 300
        crops = [
            np.zeros((48, 50, 3), dtype=np.uint8),   # group 0 (50//64=0)
            np.zeros((48, 100, 3), dtype=np.uint8),  # group 1 (100//64=1)
            np.zeros((48, 200, 3), dtype=np.uint8),  # group 3 (200//64=3)
            np.zeros((48, 300, 3), dtype=np.uint8),  # group 4 (300//64=4)
        ]
        boxes = [np.zeros((4, 2)) for _ in crops]
        batches = rec._build_batches(boxes, crops)
        # Each crop has unique width group -> 4 separate batches
        assert len(batches) == 4

    def test_group_similar_widths_together(self):
        from src.pipeline.trt_recognizer import TrtRecognizer
        rec = TrtRecognizer(batch_group_step=64)
        # Two crops with similar widths (50, 55) -> same group 0
        crops = [
            np.zeros((48, 50, 3), dtype=np.uint8),
            np.zeros((48, 55, 3), dtype=np.uint8),
        ]
        boxes = [np.zeros((4, 2)) for _ in crops]
        batches = rec._build_batches(boxes, crops)
        assert len(batches) == 1  # Both in same batch
        assert batches[0].shape[0] == 2  # Batch size 2

    def test_max_batch_splits_large_groups(self):
        from src.pipeline.trt_recognizer import TrtRecognizer
        rec = TrtRecognizer(batch_group_step=64)
        rec.MAX_BATCH = 2
        # 4 crops in same width group -> should split into 2 batches of 2
        crops = [
            np.zeros((48, 50, 3), dtype=np.uint8),
            np.zeros((48, 52, 3), dtype=np.uint8),
            np.zeros((48, 54, 3), dtype=np.uint8),
            np.zeros((48, 56, 3), dtype=np.uint8),
        ]
        boxes = [np.zeros((4, 2)) for _ in crops]
        batches = rec._build_batches(boxes, crops)
        assert len(batches) == 2
        for batch in batches:
            assert batch.shape[0] <= 2
