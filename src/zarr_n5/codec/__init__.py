from dataclasses import dataclass
from enum import IntEnum
from typing import Self, Any
import struct


class N5Mode(IntEnum):
    DEFAULT = 0
    VARLENGTH = 1
    OBJECT = 2


@dataclass
class N5BlockHeader:
    mode: N5Mode
    """Stored as >u16"""

    shape: tuple[int, ...]
    """Length stored as >u16, elements stored as >u32"""

    num_elem: int | None = None
    """Stored as >u32 if mode == VARLENGTH"""

    @classmethod
    def calc_size(cls, ndim: int, is_varlength: bool = False) -> int:
        base = 2 + 2 + 4 * ndim
        if is_varlength:
            base += 4
        return base

    def size(self) -> int:
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
        if self.num_elem is None:
            return struct.pack(fmt, *args)
        else:
            return struct.pack(fmt + "I", *args, self.num_elem)


class StructParser:
    def __init__(self, buf: bytes, endian: str = "") -> None:
        self.endian = endian
        self.buf = buf

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        fmt = self.endian + fmt
        sz = struct.calcsize(fmt)
        out = struct.unpack(fmt, self.buf[:sz])
        self.buf = self.buf[sz:]
        return out
