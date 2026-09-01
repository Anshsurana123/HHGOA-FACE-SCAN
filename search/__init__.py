"""Search and metadata extraction package."""

from search.imgbb_client import upload_to_imgbb, ImgBBError
from search.lens_client import search_google_lens, LensSearchError
from search.post_extractor import (
    fetch_post,
    is_social_media_url,
    normalize_url,
    get_social_platform,
    ALLOWED_DOMAINS,
    PostExtractionError,
)
from search.matcher import find_verified_social_post, MatcherResult

__all__ = [
    "upload_to_imgbb",
    "ImgBBError",
    "search_google_lens",
    "LensSearchError",
    "fetch_post",
    "is_social_media_url",
    "normalize_url",
    "get_social_platform",
    "ALLOWED_DOMAINS",
    "PostExtractionError",
    "find_verified_social_post",
    "MatcherResult",
]
