"""Face identification package."""

from faceid.encoder import (
    FaceEncoder,
    NoFaceFound,
    ImageReadError,
    FaceIdentificationError,
    encode_face,
    encode_face_with_meta,
    encode_all_faces,
    extract_face_crop,
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
    "encode_all_faces",
    "extract_face_crop",
    "cosine_distance",
    "faces_match",
    "DEFAULT_TOLERANCE",
]
