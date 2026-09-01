"""Unit tests for Stage 1: Face identification and embedding extraction."""

import os
import numpy as np
import pytest

from faceid.encoder import (
    FaceEncoder,
    encode_face,
    encode_face_with_meta,
    cosine_distance,
    faces_match,
    NoFaceFound,
    ImageReadError,
    DEFAULT_TOLERANCE,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_cosine_distance_properties():
    """Validates mathematical properties of cosine distance."""
    # Identity: distance to self is 0.0
    vec_a = np.random.randn(512).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)
    assert pytest.approx(cosine_distance(vec_a, vec_a), abs=1e-5) == 0.0

    # Orthogonal vectors: distance is 1.0
    vec_x = np.zeros(512, dtype=np.float32)
    vec_x[0] = 1.0
    vec_y = np.zeros(512, dtype=np.float32)
    vec_y[1] = 1.0
    assert pytest.approx(cosine_distance(vec_x, vec_y), abs=1e-5) == 1.0

    # Opposite vectors: distance is 2.0
    assert pytest.approx(cosine_distance(vec_x, -vec_x), abs=1e-5) == 2.0

    # Symmetry
    vec_b = np.random.randn(512).astype(np.float32)
    vec_b /= np.linalg.norm(vec_b)
    assert pytest.approx(cosine_distance(vec_a, vec_b), abs=1e-5) == cosine_distance(vec_b, vec_a)


def test_same_person_matches():
    """Validates that two different photos of the same person have distance < tolerance."""
    img_1a = os.path.join(FIXTURES_DIR, "person1_a.jpg")
    img_1b = os.path.join(FIXTURES_DIR, "person1_b.jpg")

    emb_1a = encode_face(img_1a)
    emb_1b = encode_face(img_1b)

    assert len(emb_1a) == 512
    assert len(emb_1b) == 512

    dist = cosine_distance(emb_1a, emb_1b)
    assert dist < DEFAULT_TOLERANCE
    assert faces_match(emb_1a, emb_1b, tol=0.35) is True


def test_different_people_do_not_match():
    """Validates that photos of different individuals exceed the match tolerance."""
    img_1a = os.path.join(FIXTURES_DIR, "person1_a.jpg")
    img_2 = os.path.join(FIXTURES_DIR, "person2.jpg")

    emb_1a = encode_face(img_1a)
    emb_2 = encode_face(img_2)

    dist = cosine_distance(emb_1a, emb_2)
    assert dist >= DEFAULT_TOLERANCE
    assert faces_match(emb_1a, emb_2, tol=0.35) is False


def test_no_face_found_raises_exception():
    """Validates that images with no faces cleanly raise NoFaceFound."""
    blank_img = os.path.join(FIXTURES_DIR, "no_face.jpg")
    with pytest.raises(NoFaceFound):
        encode_face(blank_img)


def test_invalid_image_raises_error():
    """Validates that non-existent or unreadable paths raise ImageReadError."""
    with pytest.raises(ImageReadError):
        encode_face("non_existent_file_path_12345.jpg")

    with pytest.raises(ImageReadError):
        encode_face(b"corrupted_non_image_bytes")


def test_encode_face_with_meta():
    """Validates detection metadata extraction."""
    img_path = os.path.join(FIXTURES_DIR, "person1_a.jpg")
    emb, meta = encode_face_with_meta(img_path)

    assert len(emb) == 512
    assert "bbox" in meta
    assert len(meta["bbox"]) == 4
    assert meta["total_faces_detected"] >= 1
    assert meta["det_score"] is not None and meta["det_score"] > 0.5


def test_extract_face_crop():
    """Validates that extract_face_crop returns valid JPEG bytes, 512-d embedding, and metadata."""
    from faceid import extract_face_crop
    img_path = os.path.join(FIXTURES_DIR, "person1_a.jpg")
    crop_bytes, emb, meta = extract_face_crop(img_path, margin=0.35)

    assert len(crop_bytes) > 500
    assert crop_bytes.startswith(b"\xff\xd8")  # Valid JPEG header
    assert len(emb) == 512
    assert "crop_box" in meta
    assert len(meta["crop_box"]) == 4


def test_encode_all_faces():
    """Validates that encode_all_faces returns a list of 512-d embeddings for all faces."""
    from faceid import encode_all_faces
    img_path = os.path.join(FIXTURES_DIR, "person1_a.jpg")
    embs = encode_all_faces(img_path)

    assert isinstance(embs, list)
    assert len(embs) >= 1
    for emb in embs:
        assert len(emb) == 512
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
