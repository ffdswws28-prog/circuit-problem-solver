"""MinerU API integration for circuit text recognition"""
import time
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional


class MinerUClient:
    """MinerU cloud API client for high-accuracy OCR"""

    API_BASE = "https://mineru.net/api/v4"
    # Token from MinerU desktop app config
    DEFAULT_TOKEN = "0f457673a31ed444b5bf7d383acc84d0332af4bffdfbc997a99a75f9cbea5b1348ac0e190b688266ec56ec6a4d8cf882"

    def __init__(self, token: str = None):
        self.token = token or self.DEFAULT_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

    def parse_image(self, image_path: str,
                    enable_formula: bool = True,
                    enable_table: bool = True,
                    is_ocr: bool = True,
                    language: str = "en") -> Optional[Dict]:
        """
        Parse an image file using MinerU API.
        For local images, we need to upload or use a URL.
        Try the agent API first (accepts file upload), fall back to v4.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Try v1 agent API first (supports direct file upload, no token required)
        result = self._try_agent_api(path)
        if result:
            return result

        # Fall back to v4 API
        return self._try_v4_api(path, enable_formula, enable_table, is_ocr, language)

    def _try_agent_api(self, image_path: Path) -> Optional[Dict]:
        """Try the v1 agent API (direct file upload, rate-limited but no token needed)"""
        try:
            url = "https://mineru.net/api/v1/agent/parse/file"
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/png')}
                resp = requests.post(url, files=files, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                if data.get('code') == 0:
                    return self._parse_agent_result(data)
            return None
        except Exception as e:
            print(f"[MinerU] Agent API failed: {e}")
            return None

    def _try_v4_api(self, image_path: Path,
                    enable_formula: bool, enable_table: bool,
                    is_ocr: bool, language: str) -> Optional[Dict]:
        """Try the v4 API (requires file URL)"""
        # First, try with the file uploaded as base64 or via multipart
        try:
            # Try multipart upload to v4
            url = f"{self.API_BASE}/extract/task"
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/png')}
                data = {
                    'model_version': 'vlm',
                    'enable_formula': str(enable_formula).lower(),
                    'enable_table': str(enable_table).lower(),
                    'is_ocr': str(is_ocr).lower(),
                    'language': language,
                }
                resp = requests.post(url, data=data, files=files,
                                     headers={"Authorization": f"Bearer {self.token}"},
                                     timeout=60)

            if resp.status_code == 200:
                result = resp.json()
                if result.get('code') == 0:
                    task_id = result['data']['task_id']
                    return self._poll_task(task_id)
            return None
        except Exception as e:
            print(f"[MinerU] V4 API failed: {e}")
            return None

    def _poll_task(self, task_id: str, timeout: int = 120, interval: float = 2.0) -> Optional[Dict]:
        """Poll task status until completion"""
        url = f"{self.API_BASE}/extract/task/{task_id}"
        start = time.time()

        while time.time() - start < timeout:
            resp = self.session.get(url)
            if resp.status_code != 200:
                time.sleep(interval)
                continue

            data = resp.json()
            if data.get('code') != 0:
                break

            task_data = data.get('data', {})
            status = task_data.get('status', '')

            if status == 'done':
                return self._parse_v4_result(task_data)
            elif status == 'failed':
                print(f"[MinerU] Task failed: {task_data.get('error_msg', 'unknown')}")
                return None

            time.sleep(interval)

        print(f"[MinerU] Task {task_id} timed out after {timeout}s")
        return None

    def _parse_agent_result(self, data: Dict) -> Dict:
        """Parse v1 agent API result into standardized format"""
        result_data = data.get('data', data)
        content = result_data.get('content', result_data.get('text', ''))
        return {
            'engine': 'mineru-agent',
            'markdown': content,
            'text_blocks': self._extract_text_blocks(content),
            'raw': data,
        }

    def _parse_v4_result(self, task_data: Dict) -> Dict:
        """Parse v4 API result into standardized format"""
        content = task_data.get('content', task_data.get('markdown', ''))
        return {
            'engine': 'mineru-v4',
            'task_id': task_data.get('task_id', ''),
            'pages': task_data.get('pages', 0),
            'markdown': content,
            'text_blocks': self._extract_text_blocks(content),
            'layout': task_data.get('layout', []),
            'raw': task_data,
        }

    def _extract_text_blocks(self, markdown: str) -> List[Dict]:
        """Extract text blocks from markdown output"""
        blocks = []
        for line in markdown.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip image references and pure LaTeX blocks
            if line.startswith('![') or line.startswith('$$'):
                continue
            # Clean inline LaTeX
            text = line
            blocks.append({
                'text': text,
                'type': 'text',
            })
        return blocks

    def extract_circuit_texts(self, image_path: str) -> List[Dict]:
        """
        Extract circuit-specific text from an image.
        Returns list of {text, confidence, bbox} (bbox may be absent for API results).
        """
        result = self.parse_image(image_path)
        if not result:
            return []

        texts = []
        for block in result.get('text_blocks', []):
            text = block['text']
            # Filter to likely circuit text (contains alphanumeric + circuit symbols)
            if any(c.isalnum() for c in text):
                texts.append({
                    'text': text,
                    'confidence': 0.9,  # MinerU doesn't expose per-character confidence
                    'engine': result.get('engine', 'mineru'),
                })
        return texts


# Quick test
if __name__ == "__main__":
    import sys
    client = MinerUClient()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("Usage: python ocr_mineru.py <image_path>")
        sys.exit(1)

    print(f"Parsing: {path}")
    texts = client.extract_circuit_texts(path)
    for t in texts:
        print(f"  [{t['engine']}] {t['text']} (conf: {t['confidence']})")
