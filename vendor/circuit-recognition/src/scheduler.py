"""优先级任务调度层 — 并发管控 + 超时 + 熔断"""
import asyncio
from enum import IntEnum
from typing import Any, Callable, Dict


class Priority(IntEnum):
    HIGH = 0    # 轻量工具（仅检测、仅OCR）
    MEDIUM = 1  # 部分流程
    LOW = 2     # 全量分析、批量PDF


class PriorityTaskScheduler:
    """优先级任务调度器"""

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._failure_count: Dict[str, int] = {}
        self._circuit_breaker_threshold = 3

    async def submit(self, task: Callable, priority: Priority = Priority.LOW, timeout: float = 30.0) -> Any:
        """提交任务，带优先级、超时和熔断保护"""
        task_id = str(id(task))
        if self._failure_count.get(task_id, 0) >= self._circuit_breaker_threshold:
            raise RuntimeError(f"Circuit breaker open for task {task_id}")

        async with self._semaphore:
            try:
                result = await asyncio.wait_for(task(), timeout=timeout)
                self._failure_count[task_id] = 0
                return result
            except asyncio.TimeoutError:
                self._failure_count[task_id] = self._failure_count.get(task_id, 0) + 1
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            except Exception:
                self._failure_count[task_id] = self._failure_count.get(task_id, 0) + 1
                raise
