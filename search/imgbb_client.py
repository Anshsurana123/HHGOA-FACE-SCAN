"""ImgBB API client for uploading local images to obtain public URLs."""

from __future__ import annotations

import base64
import os
import time
from typing import Union
import requests


class ImgBBError(Exception):
    """Raised when image upload to ImgBB fails."""
    pass


DEFAULT_USER_AGENT = "HH-FaceChain/1.0 (Reverse Image Search Bot; +https://github.com)"


def upload_to_imgbb(
    image_input: Union[str, bytes],
    api_key: str | None = None,
    timeout: int = 10,
    max_retries: int = 1,
) -> str:
    """
    Uploads an image to ImgBB and returns a public URL.

    Args:
        image_input: Path to local image file or raw image bytes.
        api_key: ImgBB API key (defaults to IMGBB_KEY env var).
        timeout: Request timeout in seconds (default 10s).
        max_retries: Maximum retry attempts on failure (default 1).

    Returns:
        str: Publicly accessible URL for the uploaded image.

    Raises:
        ImgBBError: If upload fails or API key is missing.
    """
    key = api_key or os.getenv("IMGBB_KEY")
    if not key:
        raise ImgBBError("Missing ImgBB API key. Set IMGBB_KEY environment variable.")

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise ImgBBError(f"Image file does not exist: {image_input}")
        with open(image_input, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(image_input, (bytes, bytearray)):
        raw_bytes = bytes(image_input)
    else:
        raise ImgBBError(f"Unsupported image input type: {type(image_input)}")

    b64_image = base64.b64encode(raw_bytes).decode("utf-8")
    upload_url = "https://api.imgbb.com/1/upload"
    payload = {"key": key, "image": b64_image}
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                upload_url,
                data=payload,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "data" in data and "url" in data["data"]:
                    return data["data"]["url"]
                raise ImgBBError(f"ImgBB returned unexpected response: {data}")
            
            error_msg = response.text
            try:
                err_json = response.json()
                if "error" in err_json and "message" in err_json["error"]:
                    error_msg = err_json["error"]["message"]
            except Exception:
                pass
            raise ImgBBError(f"ImgBB API error (HTTP {response.status_code}): {error_msg}")

        except (requests.RequestException, ImgBBError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise ImgBBError(f"Failed to upload image to ImgBB after {max_retries + 1} attempt(s): {exc}") from exc

    raise ImgBBError(f"ImgBB upload failed: {last_error}")
