"""结构化错误码体系"""
from enum import Enum


class ErrorCode(Enum):
    # 基础错误 1xxx
    FILE_NOT_FOUND = ("E1001", "文件不存在", 404)
    FILE_FORMAT_NOT_SUPPORTED = ("E1002", "文件格式不支持（仅支持JPG/PNG/BMP/PDF）", 415)
    PARAM_ERROR = ("E1003", "输入参数校验失败", 400)

    # GPU/显存错误 2xxx
    GPU_MEMORY_INSUFFICIENT = ("E2001", "GPU显存不足", 500)
    GPU_DEVICE_NOT_FOUND = ("E2002", "未检测到NVIDIA GPU", 500)
    INFERENCE_PROCESS_CRASH = ("E2003", "推理进程崩溃", 500)

    # 模型推理错误 3xxx
    YOLO_DETECTION_FAILED = ("E3001", "元器件检测失败", 500)
    SAM_SEGMENT_FAILED = ("E3002", "元件分割失败", 500)
    OCR_RECOGNIZE_FAILED = ("E3003", "文字识别失败", 500)
    NO_VALID_COMPONENTS = ("E3004", "未检测到有效元器件", 400)

    # 系统错误 4xxx
    CACHE_WRITE_FAILED = ("E4001", "缓存写入失败", 500)
    TIMEOUT_ERROR = ("E4002", "请求超时", 408)

    # 兜底
    UNKNOWN_ERROR = ("E9999", "未知系统异常", 500)

    @property
    def code(self):
        return self.value[0]

    @property
    def message(self):
        return self.value[1]

    @property
    def http_status(self):
        return self.value[2]


class CircuitRecognitionError(Exception):
    """统一异常封装，携带结构化错误码"""

    def __init__(self, error_code: ErrorCode, detail: str = None):
        self.error_code = error_code
        self.detail = detail
        super().__init__(f"[{error_code.code}] {error_code.message}" + (f": {detail}" if detail else ""))
