"""Test MinerU API connectivity"""
import sys, time, json, requests
from pathlib import Path

image_path = r'D:\ai\tmp\test_circuit_mineru.png'
token = '0f457673a31ed444b5bf7d383acc84d0332af4bffdfbc997a99a75f9cbea5b1348ac0e190b688266ec56ec6a4d8cf882'

# Create test image if needed
if not Path(image_path).exists():
    import cv2, numpy as np
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'R1 10k', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, 'C2 100uF', (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, 'U3 LM358', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.imwrite(image_path, img)
    print('Test image created')

# Try v1 agent API
print('=== v1 Agent API (file upload, no auth) ===')
try:
    url = 'https://mineru.net/api/v1/agent/parse/file'
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/png')}
        resp = requests.post(url, files=files, timeout=60)
    print(f'Status: {resp.status_code}')
    text = resp.text[:600]
    print(f'Response: {text}')
except Exception as e:
    print(f'v1 Error: {e}')

print()

# Try v4 API with multipart upload
print('=== v4 API (multipart + token) ===')
try:
    url = 'https://mineru.net/api/v4/extract/task'
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/png')}
        data = {
            'model_version': 'vlm',
            'enable_formula': 'true',
            'is_ocr': 'true',
            'language': 'en',
        }
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.post(url, data=data, files=files, headers=headers, timeout=60)
    print(f'Status: {resp.status_code}')
    result = resp.json()
    print(f'Code: {result.get("code")}')
    print(f'Msg: {result.get("msg")}')
    task_data = result.get('data', {})

    if result.get('code') == 0 and task_data.get('task_id'):
        task_id = task_data['task_id']
        print(f'Task ID: {task_id}')
        print('Polling...')

        poll_url = f'https://mineru.net/api/v4/extract/task/{task_id}'
        for i in range(30):
            time.sleep(2)
            pr = requests.get(poll_url, headers=headers, timeout=30)
            pd = pr.json()
            status = pd.get('data', {}).get('state', '')
            progress = pd.get('data', {}).get('progress', 0)
            if i < 5 or status == 'done':
                print(f'  #{i+1}: status={status}, progress={progress}')
            if status in ('done', 'success', 'failed'):
                content = pd['data'].get('content', pd['data'].get('markdown', ''))
                if content:
                    print(f'\n=== Result ===')
                    print(content[:1500])
                else:
                    print(f'\nFull response keys: {list(pd["data"].keys())}')
                break
    else:
        print(f'Full response: {json.dumps(result, indent=2)[:1000]}')
except Exception as e:
    print(f'v4 Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
