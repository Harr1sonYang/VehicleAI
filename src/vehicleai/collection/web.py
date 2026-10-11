"""Wikimedia Commons candidate discovery, validation, and metadata helpers."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

PROVIDER_NAME = "wikimedia_commons"
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def _phash(image: Image.Image) -> str:
    """Return a conventional 64-bit DCT perceptual hash as 16 hex digits."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    resized = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequencies = cv2.dct(resized)[:8, :8]
    median = float(np.median(low_frequencies.flatten()[1:]))
    bits = (low_frequencies > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _plain_text(value: Any) -> str | None:
    if not value:
        return None
    raw = value.get("value") if isinstance(value, dict) else value
    if raw is None:
        return None
    return html.unescape(re.sub(r"<[^>]+>", "", str(raw))).strip() or None


@dataclass(frozen=True)
class ImageInspection:
    sha256: str
    phash: str
    width: int
    height: int
    extension: str
    mime_type: str


@dataclass(frozen=True)
class SearchResult:
    page_id: int
    title: str
    source_url: str
    download_url: str
    source_page_url: str
    mime_type: str | None
    provider_metadata: dict[str, Any]


def inspect_image_bytes(content: bytes, *, min_width: int, min_height: int, min_pixels: int) -> ImageInspection:
    """Decode an image, reject unsupported/low-resolution content, and hash it."""
    digest = hashlib.sha256(content).hexdigest()
    try:
        with Image.open(io.BytesIO(content)) as verify_image:
            verify_image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            image_format = (image.format or "").upper()
            extension = SUPPORTED_FORMATS.get(image_format)
            if extension is None:
                raise ValueError(f"Unsupported image format: {image_format or 'unknown'}")
            width, height = image.size
            if width < min_width or height < min_height or width * height < min_pixels:
                raise ValueError(f"Image is too small: {width}x{height}")
            perceptual_hash = _phash(image)
            mime_type = Image.MIME.get(image_format, "application/octet-stream")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Corrupted or undecodable image: {error}") from error
    return ImageInspection(digest, perceptual_hash, width, height, extension, mime_type)


def build_candidate_filename(source: str, filename_hint: str, inspection: ImageInspection) -> str:
    for value, label in ((source, "source"), (filename_hint, "filename_hint")):
        if not re.fullmatch(r"[a-z0-9]+", value):
            raise ValueError(f"{label} must contain lowercase ASCII letters and digits only: {value}")
    return f"{source}_{filename_hint}_{inspection.sha256[:12]}{inspection.extension}"


class WikimediaCommonsProvider:
    """Small, sequential client for the documented Wikimedia Commons API."""

    def __init__(self, endpoint: str, user_agent: str, *, delay_seconds: float = 1.0, timeout_seconds: float = 30.0, thumbnail_width: int = 1600) -> None:
        self.endpoint = endpoint
        self.delay_seconds = max(0.0, delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.thumbnail_width = thumbnail_width
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Api-User-Agent": user_agent})
        self._last_request_at = 0.0

    def _wait(self) -> None:
        remaining = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _get(self, url: str, *, params: dict[str, Any] | None = None, stream: bool = False) -> requests.Response:
        for attempt in range(4):
            self._wait()
            response = self.session.get(url, params=params, stream=stream, timeout=self.timeout_seconds)
            self._last_request_at = time.monotonic()
            if response.status_code not in {429, 503}:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(max(self.delay_seconds, delay))
        response.raise_for_status()
        raise RuntimeError("Wikimedia request failed after retries")

    def search(self, query: str, max_results: int) -> Iterator[SearchResult]:
        continuation: dict[str, Any] = {}
        yielded = 0
        while yielded < max_results:
            params: dict[str, Any] = {
                "action": "query", "format": "json", "formatversion": 2,
                "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": 6, "gsrlimit": min(50, max_results - yielded),
                "prop": "imageinfo", "iiprop": "url|size|mime|mediatype|extmetadata",
                "iiurlwidth": self.thumbnail_width, "iiextmetadatalanguage": "en",
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription|UsageTerms",
                **continuation,
            }
            payload = self._get(self.endpoint, params=params).json()
            if "error" in payload:
                raise RuntimeError(f"Wikimedia API error: {payload['error']}")
            pages = payload.get("query", {}).get("pages", [])
            for page in pages:
                image_info = (page.get("imageinfo") or [{}])[0]
                source_url = image_info.get("url")
                download_url = image_info.get("thumburl") or source_url
                if not source_url or not download_url or image_info.get("mediatype") != "BITMAP":
                    continue
                ext = image_info.get("extmetadata") or {}
                license_name = _plain_text(ext.get("LicenseShortName"))
                license_url = _plain_text(ext.get("LicenseUrl"))
                if not license_url and license_name and license_name.casefold() == "public domain":
                    license_url = "https://commons.wikimedia.org/wiki/Commons:Public_domain"
                metadata = {
                    "license": license_name,
                    "license_url": license_url,
                    "artist": _plain_text(ext.get("Artist")),
                    "credit": _plain_text(ext.get("Credit")),
                    "description": _plain_text(ext.get("ImageDescription")),
                    "usage_terms": _plain_text(ext.get("UsageTerms")),
                }
                yield SearchResult(
                    page_id=int(page["pageid"]), title=str(page["title"]),
                    source_url=str(source_url), download_url=str(download_url),
                    source_page_url=f"https://commons.wikimedia.org/?curid={page['pageid']}",
                    mime_type=image_info.get("mime"), provider_metadata=metadata,
                )
                yielded += 1
                if yielded >= max_results:
                    return
            continuation = payload.get("continue") or {}
            if not continuation or not pages:
                return

    def search_count(self, query: str) -> int:
        """Return the provider's raw search hit count without downloading media."""
        payload = self._get(self.endpoint, params={
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "list": "search",
            "srsearch": f"{query} filetype:bitmap",
            "srnamespace": 6,
            "srlimit": 1,
            "srinfo": "totalhits",
            "srprop": "",
        }).json()
        if "error" in payload:
            raise RuntimeError(f"Wikimedia API error: {payload['error']}")
        return int(payload.get("query", {}).get("searchinfo", {}).get("totalhits", 0))

    def download(self, url: str, max_bytes: int) -> bytes:
        response = self._get(url, stream=True)
        advertised = int(response.headers.get("Content-Length", "0") or 0)
        if advertised > max_bytes:
            raise ValueError(f"Download exceeds maximum size: {advertised} bytes")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Download exceeds maximum size: {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def load_existing_hashes(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    perceptual: dict[str, str] = {}
    if not root.exists():
        return exact, perceptual
    for metadata_path in root.rglob("metadata.jsonl"):
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                filename = str(record.get("filename", ""))
                if record.get("sha256") and filename:
                    exact[str(record["sha256"])] = filename
                if record.get("phash") and filename:
                    perceptual[str(record["phash"])] = filename
    return exact, perceptual


def find_near_duplicates(phash: str, known: dict[str, str], max_distance: int) -> list[str]:
    candidate = int(phash, 16)
    return sorted(filename for known_hash, filename in known.items() if (candidate ^ int(known_hash, 16)).bit_count() <= max_distance)


def append_metadata(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
