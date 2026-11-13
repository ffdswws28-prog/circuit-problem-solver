"""E2E test: EasyOCR + Circuit Text Corrector"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import easyocr
from src.circuit_text_corrector import CircuitTextCorrector

# Build test image with circuit-specific text
img = np.ones((500, 700, 3), dtype=np.uint8) * 255
tests = [
    ("R1 10k", (50, 80)),
    ("R2 4.7k", (50, 140)),
    ("C1 100uF", (50, 200)),
    ("C2 2200uF 25V", (50, 260)),
    ("U1 LM358", (300, 80)),
    ("D1 1N4148", (300, 140)),
    ("Q1 2N2222", (300, 200)),
    ("VCC", (300, 260)),
    ("GND", (300, 320)),
    ("R3 1M", (300, 380)),
    ("C3 10nF", (50, 340)),
    ("L1 100uH", (50, 400)),
]

for text, pos in tests:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

# Load EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True)
corrector = CircuitTextCorrector()

# Run OCR
print("Running OCR...")
raw_results = reader.readtext(img)
ocr_items = [{'text': r[1], 'confidence': r[2]} for r in raw_results]

# Apply correction
corrected = corrector.correct(ocr_items)

# Display results
print()
print(f"{'Raw OCR':20s} | {'Corrected':20s} | Conf  | Status")
print("-" * 70)

expected_map = {
    'R1': 'R1', '10k': '10k', 'R2': 'R2', '4.7k': '4.7k',
    'C1': 'C1', '100uF': '100μF', 'C2': 'C2', '2200uF': '2200μF',
    '25V': '25V', 'U1': 'U1', 'LM358': 'LM358', 'D1': 'D1',
    '1N4148': '1N4148', 'Q1': 'Q1', '2N2222': '2N2222',
    'VCC': 'VCC', 'GND': 'GND', 'R3': 'R3', '1M': '1M',
    'C3': 'C3', '10nF': '10nF', 'L1': 'L1', '100uH': '100μH',
}

found = 0
missed = 0
for item in corrected:
    raw = item.get('original') or item['text']
    corr = item['text']
    status = ''

    # Check if this text matches any expected
    for exp_key, exp_val in expected_map.items():
        if exp_val == corr or exp_key in raw :
            status = 'OK'
            found += 1
            break
    if not status:
        missed += 1

    print(f"{raw:20s} | {corr:20s} | {item['confidence']:.2f}   | {status}")

print("-" * 70)
print(f"Found: {found}, Missed: {missed}")
