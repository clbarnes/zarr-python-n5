"""
Utilities for parsing, representing, and converting N5 metadata.
"""

from __future__ import annotations
from copy import deepcopy
import itertools
from typing import Any, TYPE_CHECKING, Self
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.dtype import ZDType
from zarr.core import dtype as zdt
from zarr.core.metadata.v3 import RegularChunkGridMetadata
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import blosc
from zarr.codecs import GzipCodec, ZstdCodec

from .util import N5Mode

if TYPE_CHECKING:
    from typing import Self
    from zarr.core.common import JSON

__all__ = ["N5GroupMetadata", "N5ArrayMetadata", "COMPATIBLE_DATA_TYPES"]

COMPATIBLE_DATA_TYPES: dict[str, tuple[ZDType, int]] = {
    "uint8": (zdt.UInt8(), 1),
    "uint16": (zdt.UInt16(), 2),
    "uint32": (zdt.UInt32(), 4),
    "uint64": (zdt.UInt64(), 8),
    "int8": (zdt.Int8(), 1),
    "int16": (zdt.Int16(), 2),
    "int32": (zdt.Int32(), 4),
    "int64": (zdt.Int64(), 8),
    "float32": (zdt.Float32(), 4),
    "float64": (zdt.Float64(), 8),
}
"""Data types which exist in both Zarr and N5.

Maps to the Zarr data type and item size."""


class N5GroupMetadata:
    def __init__(
        self, n5: str | None = None, attrs: dict[str, JSON] | None = None
    ) -> None:
        self.n5: str | None = n5
        self.attributes: dict[str, Any] = attrs or dict()

    def to_jso(self) -> dict[str, JSON]:
        out = deepcopy(self.attributes)
        if self.n5 is not None:
            out["n5"] = self.n5
        return out

    def is_root(self):
        return self.n5 is not None

    @classmethod
    def from_jso(cls, jso: dict[str, JSON]) -> Self:
        n5 = jso.pop("n5", None)
        if n5 is not None and not isinstance(n5, str):
            raise ValueError("n5 attribute is not a string")
        return cls(n5, jso)

    def to_zarr(self):
        attrs = deepcopy(self.attributes)
        attrs["_n5"] = self.to_jso()
        return GroupMetadata(attrs)


class N5ArrayMetadata(N5GroupMetadata):
    def __init__(
        self,
        dimensions: list[int],
        block_size: list[int],
        data_type: str,
        compression: dict[str, Any],
        n5: str | None = None,
        attrs: dict[str, JSON] | None = None,
    ):
        super().__init__(n5, attrs)
        if len(dimensions) != len(block_size):
            raise ValueError(
                f"dimensions {dimensions} and block size {block_size} must have same dimensionality"
            )

        if any(not is_nonzero_int(s) for s in itertools.chain(dimensions, block_size)):
            raise ValueError("dimensions and block size must be positive integers")

        self.dimensions = dimensions
        self.block_size = block_size
        self.data_type = data_type
        ctype = compression.get("type")
        if not isinstance(ctype, str):
            raise ValueError(f"compression must have a string type, got {ctype}")
        self.compression = compression

    def to_jso(self) -> dict[str, JSON]:
        jso = super().to_jso()
        jso["dimensions"] = self.dimensions
        jso["blockSize"] = self.block_size
        jso["dataType"] = self.data_type
        jso["compression"] = self.compression
        return jso

    @classmethod
    def from_group(cls, grp: N5GroupMetadata) -> Self:
        attrs = deepcopy(grp.attributes)
        dimensions = attrs.pop("dimensions")
        block_size = attrs.pop("blockSize")
        data_type = attrs.pop("dataType")
        compression = attrs.pop("compression")
        return cls(dimensions, block_size, data_type, compression, grp.n5, attrs)

    @classmethod
    def from_jso(cls, jso: dict[str, JSON]) -> Self:
        grp = super().from_jso(jso)
        return cls.from_group(grp)

    def to_zarr(self, mode: N5Mode = N5Mode.DEFAULT):
        from .codec.default import N5DefaultCodec

        if mode != N5Mode.DEFAULT:
            raise NotImplementedError("Only default-mode N5 is supported")
        compressor = self._to_zarr_codec()
        attrs = deepcopy(self.attributes)
        attrs["_n5"] = self.to_jso()
        return ArrayV3Metadata(
            shape=self.dimensions,
            data_type=COMPATIBLE_DATA_TYPES[self.data_type][0],
            chunk_grid=RegularChunkGridMetadata(chunk_shape=tuple(self.block_size)),
            chunk_key_encoding=V2ChunkKeyEncoding("/"),
            fill_value=0,
            dimension_names=None,
            codecs=[
                N5DefaultCodec.from_compressor(
                    len(self.dimensions),
                    compressor,
                ),
            ],
            attributes=attrs,
        )

    def _to_zarr_codec(self) -> BytesBytesCodec | None:
        tp = self.compression.get("type")
        match tp:
            case "raw":
                return None
            case "blosc":
                item_size = COMPATIBLE_DATA_TYPES[self.data_type][1]
                return parse_blosc(self.compression, item_size)
            case "gzip":
                return parse_gzip(self.compression)
            case "zstd":
                return parse_zstd(self.compression)
            case _:
                raise ValueError(f"unsupported codec with type {tp}")


def is_nonzero_int(n) -> bool:
    if not isinstance(n, int):
        return False
    return n > 0


def parse_blosc(d: dict[str, JSON], typesize: int | None) -> blosc.BloscCodec:
    cname = d.get("cname", "blosclz")
    clevel = d.get("clevel", 6)
    blocksize = d.get("blocksize", 0)
    shuffle_int = d.get("shuffle", 0)
    shuffle = blosc.BloscShuffle.from_int(shuffle_int)  # type: ignore
    return blosc.BloscCodec(
        typesize=typesize,
        cname=cname,  # type: ignore
        clevel=clevel,  # type:ignore
        blocksize=blocksize,  # type:ignore
        shuffle=shuffle,
    )


def parse_gzip(d: dict[str, JSON]) -> GzipCodec:
    level = d.get("level", -1)
    if level == -1:
        return GzipCodec()
    else:
        level = int(level)  # type: ignore
    return GzipCodec(level=level)


def parse_zstd(d: dict[str, JSON]) -> ZstdCodec:
    level = d.get("level", 3)
    return ZstdCodec(level=level)  # type: ignore
