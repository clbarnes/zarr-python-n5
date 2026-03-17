from zarr.abc.store import (
    ByteRequest,
    RangeByteRequest,
    OffsetByteRequest,
    SuffixByteRequest,
)
from .constants import ZARR_V3_METADATA_KEY


def slice_buf(b: bytes, byte_range: ByteRequest | None = None) -> bytes:
    if byte_range is None:
        pass
    elif isinstance(byte_range, RangeByteRequest):
        b = b[byte_range.start : byte_range.end]
    elif isinstance(byte_range, OffsetByteRequest):
        b = b[byte_range:]
    elif isinstance(byte_range, SuffixByteRequest):
        b = b[-byte_range.suffix :]

    return b


def is_metadata(key: str):
    return key.split("/")[-1] == ZARR_V3_METADATA_KEY
