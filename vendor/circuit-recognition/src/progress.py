"""进度推送器 — 分阶段进度通知"""
from enum import Enum
from typing import Callable, Optional


class PipelineStage(Enum):
    PREPROCESS = ("preprocess", "图片预处理", 5)
    YOLO_DETECT = ("yolo_detect", "YOLO 目标检测", 20)
    SAM_SEGMENT = ("sam_segment", "SAM 按需分割", 30)
    OCR_RECOGNIZE = ("ocr_recognize", "OCR 文字识别", 25)
    TOPOLOGY_ANALYZE = ("topology_analyze", "拓扑分析", 15)
    FUSION = ("fusion", "结果融合+输出", 5)


class ProgressNotifier:
    """进度推送，每阶段完成后推送进度百分比"""

    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self._progress = 0

    @property
    def progress(self) -> int:
        return self._progress

    def update(self, stage: PipelineStage):
        """更新进度到指定阶段"""
        self._progress += stage.value[2]
        self._progress = min(self._progress, 100)
        if self.callback:
            self.callback(self._progress, stage.value[1])

    def reset(self):
        self._progress = 0
