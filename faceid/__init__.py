"""Face identification package."""

from faceid.encoder import (
    FaceEncoder,
    NoFaceFound,
    ImageReadError,
    FaceIdentificationError,
    encode_face,
    encode_face_with_meta,
    cosine_distance,
    faces_match,
    DEFAULT_TOLERANCE,
)

__all__ = [
    "FaceEncoder",
    "NoFaceFound",
    "ImageReadError",
    "FaceIdentificationError",
    "encode_face",
    "encode_face_with_meta",
    "cosine_distance",
    "faces_match",
    "DEFAULT_TOLERANCE",
]
