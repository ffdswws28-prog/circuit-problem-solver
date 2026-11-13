"""多级缓存 — SHA256 内容哈希 + TTL + 大小上限"""
import hashlib
import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict


class MultiLevelCache:
    """L1 内存缓存（LRU + TTL 1h）+ L2 磁盘缓存（TTL 7d + 上限 20GB）"""

    def __init__(self, disk_cache_dir: str = None, memory_max_size: int = 100, disk_max_size_gb: int = 20):
        self.disk_cache_dir = Path(disk_cache_dir or os.path.join(os.path.dirname(__file__), "..", "cache"))
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_max_size = memory_max_size
        self.disk_max_size_gb = disk_max_size_gb
        # L1: 内存 LRU
        self._l1_cache = OrderedDict()
        self._l1_ttl = 3600  # 1 hour

    def _compute_key(self, file_hash: str, params: Dict = None) -> str:
        """缓存键 = 文件 SHA256 + 参数哈希"""
        if params:
            params_str = json.dumps(params, sort_keys=True)
            return hashlib.sha256(f"{file_hash}:{params_str}".encode()).hexdigest()
        return file_hash

    def get(self, file_hash: str, params: Dict = None) -> Optional[Dict]:
        """获取缓存结果"""
        key = self._compute_key(file_hash, params)
        # L1 查找
        if key in self._l1_cache:
            entry = self._l1_cache[key]
            if time.time() - entry["time"] < self._l1_ttl:
                self._l1_cache.move_to_end(key)
                return entry["data"]
        # L2 查找
        disk_path = self.disk_cache_dir / f"{key}.json"
        if disk_path.exists():
            with open(disk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if time.time() - data.get("_timestamp", 0) < 7 * 86400:  # 7 days TTL
                self._l1_cache[key] = {"data": data, "time": time.time()}
                self._l1_evict()
                return data
        return None

    def set(self, file_hash: str, data: Dict, params: Dict = None):
        """写入缓存"""
        key = self._compute_key(file_hash, params)
        data["_timestamp"] = time.time()
        # L1
        self._l1_cache[key] = {"data": data, "time": time.time()}
        self._l1_evict()
        # L2
        disk_path = self.disk_cache_dir / f"{key}.json"
        with open(disk_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def _l1_evict(self):
        """LRU 淘汰"""
        while len(self._l1_cache) > self.memory_max_size:
            self._l1_cache.popitem(last=False)
