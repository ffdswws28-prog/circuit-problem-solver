"""Circuit-specific OCR text correction layer

Fixes common OCR errors in circuit diagrams:
- O->0 in numeric contexts (1OOuF -> 100uF)
- I->1 in part numbers (IN4148 -> 1N4148)
- Component label validation (R1, C2, U3, etc.)
"""
import re
from typing import List, Dict


class CircuitTextCorrector:
    """Post-process OCR results with circuit-domain knowledge"""

    # Common circuit component prefixes
    COMPONENT_PREFIXES = {'R', 'C', 'L', 'D', 'Q', 'U', 'J', 'P', 'T', 'X', 'Y', 'K', 'F', 'M'}

    # Common circuit value units
    VALUE_UNITS = {'Ω', 'k', 'M', 'μ', 'F', 'H', 'V', 'A', 'W', 'p', 'n', 'm', 'u', 'f', 'h'}

    # Known IC part number patterns
    IC_PATTERNS = [
        r'^[1I][nN]\d+',       # 1N4148, 1N4007 (diodes)
        r'^[1I][lL]\d+',       # 1L...
        r'^[lL][mM]\d+',       # LM358, LM324 (op-amps)
        r'^[nN][eE]\d+',       # NE555 (timers)
        r'^[tT][lL]\d+',       # TL072, TL081
        r'^[uU][aA]\d+',       # UA741
        r'^[mM][aA][xX]\d+',   # MAX232
        r'^[aA][dD]\d+',       # AD620
        r'^[lL][tT]\d+',       # LT1013
        r'^[oO][pP]\d+',       # OP07
    ]

    def __init__(self):
        pass

    def correct(self, texts: List[Dict]) -> List[Dict]:
        """Apply all corrections to OCR results"""
        corrected = []
        for item in texts:
            text = item.get('text', '')
            confidence = item.get('confidence', 0.0)

            original = text
            text = self._fix_component_label(text)
            text = self._fix_numeric_context(text)
            text = self._fix_part_number(text)
            text = self._fix_unit_value(text)
            text = self._validate(text)

            corrected.append({
                **item,
                'text': text,
                'original': original if original != text else None,
                'confidence': confidence,
            })
        return corrected

    def _fix_component_label(self, text: str) -> str:
        """Fix component labels: ensure prefix+number pattern
        Example: R1, Cl -> C1 (but careful with real 'Cl' like in 'clear')
        """
        # Check for single-letter prefix followed by digits
        # Common mistake: C gets OCR'd as c, O as 0, etc.
        for prefix in self.COMPONENT_PREFIXES:
            pattern = re.compile(rf'\b[{prefix.lower()}{prefix}]\d+\b', re.IGNORECASE)
            text = pattern.sub(
                lambda m: prefix + m.group()[1:],
                text
            )
        return text

    def _fix_numeric_context(self, text: str) -> str:
        """Fix O->0 and l/I->1 in numeric contexts

        Rules:
        - 'O' in numeric context -> '0' (e.g., 1OOuF -> 100uF)
        - letter 'l' in numeric context -> '1' (e.g., l0k -> 10k)
        """
        # Replace O with 0 after a digit (covers 1O, 1OO, etc.)
        text = re.sub(r'(\d)(O+)(\d)', lambda m: m.group(1) + '0' * len(m.group(2)) + m.group(3), text)
        text = re.sub(r'(\d)(O+)([kK][Ω]?|[MμunpfFHAVW])',
                      lambda m: m.group(1) + '0' * len(m.group(2)) + m.group(3), text)

        # l between digits or leading l+d -> 1
        text = re.sub(r'(\d)(l+)(\d)', lambda m: m.group(1) + '1' * len(m.group(2)) + m.group(3), text)
        text = re.sub(r'\b(l)(\d)', r'1\g<2>', text)

        return text

    def _fix_part_number(self, text: str) -> str:
        """Fix common IC part number OCR errors

        The most common errors:
        - I -> 1 at start (IN4148 -> 1N4148)
        - O -> 0 in numbers
        """
        for pattern in self.IC_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Fix I/i -> 1 at start
                prefix = match.group()
                if prefix[0] in ('I', 'i'):
                    text = '1' + text[1:]
                return text
        return text

    def _fix_unit_value(self, text: str) -> str:
        """Fix unit/value notation

        Common patterns in circuits:
        - 10k -> 10kΩ (implicit Ω)
        - 100uF -> 100μF (u -> μ)
        """
        # u -> μ (micro prefix + unit context)
        text = re.sub(r'(\d+)u([FfHh])', r'\g<1>μ\g<2>', text)

        # Standardize k/M/G multipliers
        # Already handled by OCR if case is correct

        return text

    def _validate(self, text: str) -> str:
        """Final validation pass"""
        # Strip whitespace
        text = text.strip()

        # Remove clearly non-circuit artifacts
        # (keep alphanumeric + common circuit symbols)
        allowed = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzΩkMμunpfFHAVW/-+.() ')
        text = ''.join(c for c in text if c in allowed)
        return text


def demo():
    """Quick demonstration of correction"""
    corrector = CircuitTextCorrector()

    test_cases = [
        # (input, expected)
        ("1OOuF", "100μF"),
        ("IN4148", "1N4148"),
        ("1Ok", "10k"),
        ("R1", "R1"),
        ("LM358", "LM358"),
        ("Cl", "Cl"),  # Ambiguous: could be C1 but could be 'Cl'
        ("VCC", "VCC"),
        ("l0k", "10k"),
        ("U3", "U3"),
        ("D4", "D4"),
        ("4k7", "4k7"),  # European notation
        ("100nF", "100nF"),
        ("2200uF", "2200μF"),
    ]

    print("Circuit Text Corrector Demo")
    print("-" * 50)
    for inp, expected in test_cases:
        result = corrector.correct([{'text': inp, 'confidence': 0.5}])
        out = result[0]['text']
        status = "OK" if out == expected else f"MISMATCH (expected '{expected}')"
        print(f"  {inp:12s} -> {out:12s}  {status}")


if __name__ == "__main__":
    demo()
