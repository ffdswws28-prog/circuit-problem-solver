"""GPU 显存管理 — 配额制管控 + 峰值监控 + 崩溃自动重启"""
import subprocess
import time
import threading
from typing import Dict


RESOURCE_CONFIG = {
    "yolo": {"cmd": ["python", "src/pipeline/yolo_detector.py"], "memory": 1800},
    "ocr":  {"cmd": ["python", "src/pipeline/ocr_engine.py"],    "memory": 1200},
    "sam":  {"cmd": ["python", "src/pipeline/sam_segmentor.py"], "memory": 2000},
}
# 常驻占用: 1800+1200=3000MB (YOLO+OCR)
# SAM 按需: +2000MB，总计 5000MB
# 系统预留: 1500MB
# 可用上限: 6500MB


class GPUResourceManager:
    """显存配额管理 + 进程保活监控"""

    def __init__(self, total_memory_mb: int = 8192, reserved_mb: int = 1500):
        self.total_memory = total_memory_mb
        self.reserved = reserved_mb
        self.available = total_memory_mb - reserved_mb
        self.allocations: Dict[str, int] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._monitor_thread = None

    def allocate(self, name: str, memory_mb: int) -> bool:
        """尝试分配显存配额"""
        current = sum(self.allocations.values())
        if current + memory_mb > self.available:
            return False
        self.allocations[name] = memory_mb
        return True

    def release(self, name: str):
        """释放显存配额"""
        self.allocations.pop(name, None)

    def start_monitor(self):
        """启动后台监控线程"""
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self):
        """每 5s 采样显存，峰值超 90% 告警"""
        while True:
            # TODO: 调用 nvidia-smi 或 pynvml 查询显存
            time.sleep(5)
