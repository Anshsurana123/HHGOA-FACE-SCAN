import os
import numpy as np
import pytest

from faceid.encoder import (
    encode_face_with_meta,
    extract_face_crop,
    extract_canonical_aligned_face,
    cosine_distance,
)
from search.ocr_extractor import extract_image_text_and_keywords

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_A1 = os.path.join(FIXTURES_DIR, "person1_a.jpg")


def test_68_and_106_landmarks_and_3d_pose():
    """Verify that 68-point 3D landmarks, 106-point 2D landmarks, and 3D pose are extracted."""
    emb, meta = encode_face_with_meta(str(SAMPLE_A1))
    assert emb.shape == (512,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    # 68 3D landmarks
    lmk68 = meta.get("landmark_3d_68")
    assert isinstance(lmk68, list)
    assert len(lmk68) == 68
    for pt in lmk68:
        assert len(pt) == 3

    # 106 2D dense landmarks
    lmk106 = meta.get("landmark_2d_106")
    assert isinstance(lmk106, list)
    assert len(lmk106) == 106
    for pt in lmk106:
        assert len(pt) == 2

    # 3D Head pose angles (pitch, yaw, roll)
    pose = meta.get("pose")
    assert isinstance(pose, list)
    assert len(pose) == 3


def test_canonical_umeyama_affine_alignment():
    """Verify Umeyama canonical alignment produces a standardized face image."""
    canon_bytes, emb, meta = extract_canonical_aligned_face(str(SAMPLE_A1), image_size=224)
    assert isinstance(canon_bytes, bytes)
    assert len(canon_bytes) > 500
    assert emb.shape == (512,)
    assert meta.get("alignment") == "umeyama_canonical"
    assert meta.get("aligned_size") == 224


def test_ocr_extractor_text_and_keywords():
    """Verify OCR extractor gracefully parses text and extracts keyword candidates."""
    res = extract_image_text_and_keywords(str(SAMPLE_A1))
    assert isinstance(res, dict)
    assert "full_text" in res
    assert "segments" in res
    assert "keywords" in res
