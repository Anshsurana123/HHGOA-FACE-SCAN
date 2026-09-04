"""Multi-modal OCR extractor module to extract text, names, and keywords from query images."""

from __future__ import annotations

import os
import re
from typing import Any
import numpy as np
import cv2

# Global lazy OCR reader instance
_READER = None


def _get_ocr_reader():
    global _READER
    if _READER is None:
        try:
            import easyocr
            _READER = easyocr.Reader(["en"], gpu=False)
        except Exception:
            _READER = None
    return _READER


def extract_image_text_and_keywords(
    image_input: str | bytes | np.ndarray,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    """
    Extracts text segments, bounding boxes, and identity keywords from an image.
    
    Returns:
        dict with:
            - full_text: string concatenation of all detected text
            - segments: list of dicts with {"text": ..., "confidence": ..., "bbox": ...}
            - keywords: list of extracted candidate name / identity search phrases
    """
    img_bgr = None
    if isinstance(image_input, np.ndarray):
        img_bgr = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str) and os.path.exists(image_input):
        img_bgr = cv2.imread(image_input)

    if img_bgr is None:
        return {"full_text": "", "segments": [], "keywords": []}

    segments: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    # 1. Try EasyOCR first
    reader = _get_ocr_reader()
    if reader is not None:
        try:
            results = reader.readtext(img_bgr)
            for bbox, text, conf in results:
                t = text.strip()
                if t and float(conf) >= min_confidence:
                    segments.append({
                        "text": t,
                        "confidence": round(float(conf), 3),
                        "bbox": [[float(p[0]), float(p[1])] for p in bbox],
                    })
                    full_text_parts.append(t)
        except Exception:
            pass

    # 2. Fallback to pytesseract if EasyOCR didn't find segments
    if not segments:
        try:
            import pytesseract
            from PIL import Image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            raw_text = pytesseract.image_to_string(pil_img)
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            for line in lines:
                segments.append({
                    "text": line,
                    "confidence": 0.8,
                    "bbox": [],
                })
                full_text_parts.append(line)
        except Exception:
            pass

    full_text = " ".join(full_text_parts)

    # 3. Extract Identity Keywords & Query Phrases
    keywords: list[str] = []
    for seg in segments:
        text = seg["text"]
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", text).strip()
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() or w.isupper() for w in words if w):
                clean_name = " ".join(words)
                if clean_name not in keywords:
                    keywords.append(clean_name)

    # Look for location keywords
    loc_part = ""
    for seg in segments:
        text_lower = seg["text"].lower()
        for loc in ["goa", "mumbai", "delhi", "pune", "bangalore", "india", "hyderabad", "chennai"]:
            if loc in text_lower:
                loc_part = loc.title()
                break
        if loc_part:
            break

    if keywords and loc_part:
        combined = f"{keywords[0]} {loc_part}"
        if combined not in keywords:
            keywords.insert(0, combined)

    return {
        "full_text": full_text,
        "segments": segments,
        "keywords": keywords,
    }
