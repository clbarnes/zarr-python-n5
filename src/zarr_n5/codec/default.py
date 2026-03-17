from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Self

from zarr.abc.codec import ArrayBytesCodec, Codec, BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.dtype.wrapper import TBaseDType, ZDType, TBaseScalar
from zarr.core.common import JSON, parse_named_configuration
from zarr.codecs import BytesCodec, Endian, TransposeCodec

from ..metadata import COMPATIBLE_DATA_TYPES

from . import N5BlockHeader, N5Mode

N5_DEFAULT_NAME = "n5_default"
ENDIAN = Endian.big


def check_valid_transpose(codec: Codec):
    if not isinstance(codec, TransposeCodec):
        raise ValueError("not transpose codec")
    if codec.order != tuple(sorted(codec.order, reverse=True)):
        raise ValueError("not a full transpose")


def check_valid_bytes(codec: Codec):
    if not isinstance(codec, BytesCodec):
        raise ValueError("not bytes codec")
    if codec.endian != ENDIAN:
        raise ValueError("bytes codec must be big-endian")


def check_valid_compressor(codec: Codec):
    if not isinstance(codec, BytesBytesCodec):
        raise ValueError("codec is not bytes-to-bytes")


CodecTuple = (
    tuple[TransposeCodec, BytesCodec]
    | tuple[TransposeCodec, BytesCodec, BytesBytesCodec]
)


@dataclass(frozen=True)
class N5DefaultCodec(ArrayBytesCodec):
    codecs: CodecTuple

    def __init__(self, *, codecs: CodecTuple) -> None:
        if not 2 <= len(codecs) <= 3:
            raise ValueError(f"expected 2-3 codecs, got {len(codecs)}")
        check_valid_transpose(codecs[0])
        check_valid_bytes(codecs[1])
        if len(codecs) > 2:
            check_valid_compressor(codecs[2])

        object.__setattr__(self, "codecs", codecs)

    @classmethod
    def from_compressor(cls, ndim: int, compressor: BytesBytesCodec | None = None):
        order = list(range(ndim))
        order.reverse()
        codecs = [TransposeCodec(order=tuple(order)), BytesCodec(endian=ENDIAN)]
        if compressor is not None:
            codecs.append(compressor)
        return cls(codecs=tuple(codecs))

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        header_length = N5BlockHeader.calc_size(chunk_spec.ndim, False)

        for c in self.codecs:
            input_byte_length = c.compute_encoded_size(input_byte_length, chunk_spec)
            chunk_spec = c.resolve_metadata(chunk_spec)

        return input_byte_length + header_length

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        for c in self.codecs:
            chunk_spec = c.resolve_metadata(chunk_spec)
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        check_valid_transpose(self.codecs[0])
        order = list(range(array_spec.ndim))
        order.reverse()
        new = deepcopy(self)
        setattr(new.codecs[0], "order", tuple(order))
        return new

    @property
    def ndim(self):
        return len(self.codecs[0].order)

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(shape) != len(self.codecs[0].order):
            raise ValueError(f"array is {len(shape)}D, codec is {self.ndim}D")
        if not dtype._zarr_v3_name not in COMPATIBLE_DATA_TYPES:
            raise ValueError(f"N5 does not support data type {dtype}")

        return super().validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        b = chunk_data.as_buffer_like()
        header = N5BlockHeader.from_bytes(b)
        offset = header.size()
        b = b[offset:]
        buf = Buffer.from_bytes(b)  # type:ignore
        reprs = [
            ArraySpec(
                header.shape,
                chunk_spec.dtype,
                chunk_spec.fill_value,
                chunk_spec.config,
                chunk_spec.prototype,
            )
        ]

        for c in self.codecs:
            reprs.append(c.resolve_metadata(reprs[-1]))
        reprs.pop()

        for c, cs in zip(reversed(self.codecs), reversed(reprs)):
            buf = await c._decode_single(buf, cs)  # type:ignore

        buf: NDBuffer

        if buf.shape == chunk_spec.shape:
            return buf

        if all(s1 < s2 for s1, s2 in zip(chunk_spec.shape, buf.shape)):
            nd = buf.as_ndarray_like()
            nd2 = nd[tuple(slice(0, s) for s in chunk_spec.shape)]  # type:ignore
            return NDBuffer.from_ndarray_like(nd2)

        out = NDBuffer.create(
            shape=chunk_spec.shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
            fill_value=chunk_spec.fill_value,
        )
        common_shape = [min(s1, s2) for s1, s2 in zip(chunk_spec.shape, buf.shape)]
        slices = tuple(slice(0, s) for s in common_shape)
        out.as_ndarray_like()[slices] = buf.as_ndarray_like()[slices]  # type:ignore

        return out

    async def _encode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        header = N5BlockHeader(N5Mode.DEFAULT, chunk_spec.shape)
        for c in self.codecs:
            chunk_data = c._encode_single(chunk_data, chunk_spec)  # type:ignore
            chunk_spec = c.resolve_metadata(chunk_spec)

        buf: Buffer = chunk_data  # type: ignore

        bio = BytesIO()
        bio.write(header.to_bytes())
        # TODO: avoid this copy?
        bio.write(buf.as_buffer_like())
        return Buffer.from_bytes(bio.getbuffer())

    @classmethod
    def from_dict(
        cls,
        data: dict[str, JSON],
    ) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, N5_DEFAULT_NAME, require_configuration=True
        )

        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(
        self,
    ) -> dict[str, JSON]:
        return {"name": N5_DEFAULT_NAME, "codecs": [c.to_dict() for c in self.codecs]}
