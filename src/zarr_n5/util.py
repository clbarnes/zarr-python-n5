"""
General utilities.
"""

from zarr.abc.store import (
    ByteRequest,
    RangeByteRequest,
    OffsetByteRequest,
    SuffixByteRequest,
)
from .constants import ZARR_V3_METADATA_KEY
from dataclasses import dataclass
from enum import IntEnum
from typing import Self, Any
import struct

__all__ = ["N5Mode", "N5BlockHeader"]


class N5Mode(IntEnum):
    """N5 block mode"""

    DEFAULT = 0
    VARLENGTH = 1
    OBJECT = 2


@dataclass
class N5BlockHeader:
    """Parsed representation of the N5 block header."""

    mode: N5Mode
    """Stored as >u16"""

    shape: tuple[int, ...]
    """Length stored as >u16, elements stored as >u32"""

    num_elem: int | None = None
    """Stored as >u32 if mode == VARLENGTH"""

    def __post_init__(self):
        if self.num_elem is not None and self.mode != N5Mode.VARLENGTH:
            raise ValueError("num_elem must be None if mode is not VARLENGTH")

    @classmethod
    def calc_size(cls, ndim: int, is_varlength: bool = False) -> int:
        """Calculate the number of bytes in an N5 block header."""
        base = 2 + 2 + 4 * ndim
        if is_varlength:
            base += 4
        return base

    def size(self) -> int:
        """Determine the number of bytes this header will take."""
        return self.calc_size(len(self.shape), self.mode == N5Mode.VARLENGTH)

    @classmethod
    def from_bytes(cls, b: bytes) -> Self:
        p = StructParser(b, ">")
        mode_num, ndim = p.unpack("HH")
        mode = N5Mode(mode_num)

        shape = p.unpack("I" * ndim)

        if mode == N5Mode.VARLENGTH:
            numel = p.unpack("I")[0]
        else:
            numel = None

        return cls(mode=mode, shape=shape, num_elem=numel)

    @property
    def ndim(self):
        return len(self.shape)

    def to_bytes(self) -> bytes:
        fmt = ">HH" + "I" * self.ndim
        args = [self.mode, self.ndim, *self.shape]
        if self.num_elem is not None:
            fmt += "I"
            args.append(self.num_elem)
        return struct.pack(fmt, *args)


class StructParser:
    def __init__(self, buf: bytes, endian: str = "") -> None:
        self.endian = endian
        self.buf = buf
        self.offset = 0

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        fmt = self.endian + fmt
        sz = struct.calcsize(fmt)
        out = struct.unpack(fmt, self.buf[self.offset : self.offset + sz])
        self.offset += sz
        return out


def slice_buf(b: bytes, byte_range: ByteRequest | None = None) -> bytes:
    """Optionally slice a byte buffer."""
    if byte_range is None:
        return b
    elif isinstance(byte_range, RangeByteRequest):
        b = b[byte_range.start : byte_range.end]
    elif isinstance(byte_range, OffsetByteRequest):
        b = b[byte_range.offset :]
    elif isinstance(byte_range, SuffixByteRequest):
        b = b[-byte_range.suffix :]

    raise TypeError(f"byte_range argument has unknown type {type(byte_range)}")


def is_zarr3_metadata(key: str):
    """Whether a key belongs to a Zarr v3 metadata object."""
    return key.split("/")[-1] == ZARR_V3_METADATA_KEY
