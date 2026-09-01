"""Face Identification and Embedding module using InsightFace buffalo_l."""

from __future__ import annotations

import os
from typing import Union
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class FaceIdentificationError(Exception):
    """Base exception for face identification errors."""
    pass


class NoFaceFound(FaceIdentificationError):
    """Raised when no face is detected in the given image."""
    pass


class ImageReadError(FaceIdentificationError):
    """Raised when the image file or bytes cannot be decoded."""
    pass


DEFAULT_TOLERANCE = float(os.getenv("FACE_MATCH_TOL", "0.35"))


class FaceEncoder:
    """Encapsulates InsightFace FaceAnalysis for detection and embedding extraction."""

    _instance: FaceEncoder | None = None

    def __init__(self, name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)):
        available_providers = []
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                available_providers.append("CUDAExecutionProvider")
        except Exception:
            pass
        available_providers.append("CPUExecutionProvider")

        self.app = FaceAnalysis(name=name, providers=available_providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.det_size = det_size

    @classmethod
    def get_instance(cls) -> FaceEncoder:
        """Singleton accessor to avoid re-loading ONNX models repeatedly."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def read_image(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Reads and validates an image from path, raw bytes, or existing numpy array."""
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ImageReadError("Empty numpy image array provided.")
            return image_input

        if isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ImageReadError("Failed to decode image from raw bytes.")
            return img

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ImageReadError(f"Image file does not exist: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ImageReadError(f"Failed to read image file: {image_input}")
            return img

        raise ImageReadError(f"Unsupported image input type: {type(image_input)}")

    def encode_face(self, image_input: Union[str, bytes, np.ndarray]) -> tuple[np.ndarray, dict]:
        """
        Detects faces in the image and returns the 512-d embedding of the largest face.
        
        Returns:
            tuple[np.ndarray, dict]: (512-d normalized embedding, metadata dict with bbox, score, etc.)
        Raises:
            NoFaceFound: If zero faces are detected.
            ImageReadError: If the image cannot be loaded.
        """
        img = self.read_image(image_input)
        faces = self.app.get(img)

        if not faces:
            raise NoFaceFound("No face detected in the input image.")

        # If multiple faces exist, pick the largest face by bounding box area: (x2-x1) * (y2-y1)
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = largest_face.normed_embedding
        if embedding is None:
            # Fallback to normalizing raw embedding
            raw = largest_face.embedding
            norm = np.linalg.norm(raw)
            if norm == 0:
                raise FaceIdentificationError("Zero-norm face embedding generated.")
            embedding = raw / norm

        metadata = {
            "bbox": [float(c) for c in largest_face.bbox],
            "det_score": float(largest_face.det_score) if hasattr(largest_face, "det_score") else None,
            "gender": int(largest_face.gender) if hasattr(largest_face, "gender") and largest_face.gender is not None else None,
            "age": int(largest_face.age) if hasattr(largest_face, "age") and largest_face.age is not None else None,
            "total_faces_detected": len(faces),
        }

        return np.asarray(embedding, dtype=np.float32), metadata


def encode_face(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Helper function returning 512-d embedding of the largest face in image."""
    encoder = FaceEncoder.get_instance()
    embedding, _ = encoder.encode_face(image_input)
    return embedding


def encode_face_with_meta(image_input: Union[str, bytes, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Helper function returning embedding + detection metadata."""
    encoder = FaceEncoder.get_instance()
    return encoder.encode_face(image_input)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine distance 1.0 - (a . b) / (||a|| * ||b||)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Bound to [-1.0, 1.0] to prevent slight numerical floating point inaccuracies
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(1.0 - cos_sim)


def faces_match(a: np.ndarray, b: np.ndarray, tol: float | None = None) -> bool:
    """
    Checks whether two face embeddings belong to the same person.
    InsightFace cosine distance threshold (default: 0.35).
    """
    if tol is None:
        tol = DEFAULT_TOLERANCE
    dist = cosine_distance(a, b)
    return dist < tol
