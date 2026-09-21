"""Minimal, dependency-free PNG/JPEG dimension reader.

Reads only the headers, so it is cheap to run on every uploaded image and never decodes pixels.
"""

import struct
from typing import NamedTuple

_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
# SOF4 / SOF8 / SOF12 are DHT / JPG / DAC markers, not frame headers.
_NOT_FRAME_MARKERS = {0xC4, 0xC8, 0xCC}


class ImageDimensions(NamedTuple):
    width: int
    height: int


def read_image_dimensions(data: bytes) -> ImageDimensions | None:
    if data[:8] == _PNG_SIGNATURE:
        return _read_png(data)
    if data[:2] == b"\xff\xd8":
        return _read_jpeg(data)
    return None


def _read_png(data: bytes) -> ImageDimensions | None:
    if len(data) < 24:
        return None
    # IHDR is always the first chunk: 8-byte signature, 4-byte length, 4-byte "IHDR", then
    # big-endian u32 width and height.
    width, height = struct.unpack_from(">II", data, 16)
    return ImageDimensions(width, height)


def _read_jpeg(data: bytes) -> ImageDimensions | None:
    offset = 2
    while offset + 9 <= len(data):
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        # SOFn markers (start of frame) carry the dimensions.
        if 0xC0 <= marker <= 0xCF and marker not in _NOT_FRAME_MARKERS:
            height, width = struct.unpack_from(">HH", data, offset + 5)
            return ImageDimensions(width, height)
        (segment_length,) = struct.unpack_from(">H", data, offset + 2)
        offset += 2 + segment_length
    return None
