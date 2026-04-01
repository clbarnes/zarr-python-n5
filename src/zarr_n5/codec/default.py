from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from zarr.abc.codec import ArrayBytesCodec, Codec, BytesBytesCodec, CodecPipeline
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer.core import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.dtype.wrapper import TBaseDType, ZDType, TBaseScalar
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.metadata.v3 import parse_codecs
from zarr.codecs import BytesCodec, Endian, TransposeCodec
from zarr.registry import get_pipeline_class

from ..metadata import COMPATIBLE_DATA_TYPES

from ..util import N5BlockHeader

N5_DEFAULT_NAME = "n5_default"
N5_ENDIAN = Endian.big


def check_valid_transpose(codec: Codec):
    if not isinstance(codec, TransposeCodec):
        raise ValueError("not transpose codec")
    if codec.order != tuple(sorted(codec.order, reverse=True)):
        raise ValueError("not a full transpose")


def check_valid_bytes(codec: Codec):
    if not isinstance(codec, BytesCodec):
        raise ValueError("not bytes codec")
    if codec.endian != N5_ENDIAN:
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

    def __init__(self, *, codecs: Iterable[Codec | dict[str, JSON]]) -> None:
        cs = parse_codecs(codecs)
        if not 2 <= len(cs) <= 3:
            raise ValueError(f"expected 2-3 codecs, got {len(cs)}")
        check_valid_transpose(cs[0])
        check_valid_bytes(cs[1])
        if len(cs) > 2:
            check_valid_compressor(cs[2])

        object.__setattr__(self, "codecs", cs)

    @property
    def codec_pipeline(self) -> CodecPipeline:
        return get_pipeline_class().from_codecs(self.codecs)

    @classmethod
    def from_compressor(cls, ndim: int, compressor: BytesBytesCodec | None = None):
        transpose = cls.make_transpose(ndim)
        endian = cls.make_bytes()
        codecs: CodecTuple
        if compressor is None:
            codecs = (transpose, endian)
        else:
            codecs = (transpose, endian, compressor)
        return cls(codecs=codecs)

    @classmethod
    def make_transpose(cls, ndim: int) -> TransposeCodec:
        order = list(range(ndim))
        return TransposeCodec(order=tuple(reversed(order)))

    @classmethod
    def make_bytes(cls) -> BytesCodec:
        return BytesCodec(endian=N5_ENDIAN)

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
        transpose = self.make_transpose(array_spec.ndim).evolve_from_array_spec(
            array_spec
        )
        endian = self.make_bytes().evolve_from_array_spec(array_spec)

        codecs: CodecTuple
        match len(self.codecs):
            case 2:
                codecs = (transpose, endian)
            case 3:
                compressor: BytesBytesCodec = self.codecs[2].evolve_from_array_spec(  # type:ignore
                    array_spec
                )
                codecs = (transpose, endian, compressor)
            case _:
                raise ValueError("unsupported number of codecs")

        return type(self)(codecs=codecs)

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
        if dtype._zarr_v3_name not in COMPATIBLE_DATA_TYPES:
            raise ValueError(f"N5 does not support data type {dtype._zarr_v3_name}")

        return super().validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        b = chunk_data.as_buffer_like()
        header = N5BlockHeader.from_bytes(b)
        offset = header.size()

        body_buf = chunk_data[offset:]
        body_nd = chunk_spec.prototype.nd_buffer.empty(
            header.shape, chunk_spec.dtype.to_native_dtype(), chunk_spec.order
        )
        if header.shape == chunk_spec.shape:
            body_spec = chunk_spec
            all_eq = True
        else:
            body_spec = ArraySpec(
                header.shape,
                chunk_spec.dtype,
                chunk_spec.fill_value,
                chunk_spec.config,
                chunk_spec.prototype,
            )
            all_eq = False
        maybe_body_nd, *_ = await self.codec_pipeline.decode([(body_buf, body_spec)])
        # TODO: use codec_pipeline.read() instead; this should avoid the copy for truncated-block cases
        if maybe_body_nd is None:
            raise RuntimeError("unexpected nullish buffer")
        else:
            body_nd = maybe_body_nd

        if all_eq:
            # don't need to truncate or pad
            return body_nd

        # whether we can get the chunk we want by trimming down the N5 block body
        can_trim = True

        min_shape = []
        slice_lst = []
        for hs, cs in zip(header.shape, chunk_spec.shape):
            if cs > hs:
                # requested chunk is larger than the N5 block in some dimension
                can_trim = False
            min_len = min(hs, cs)
            min_shape.append(min_len)
            slice_lst.append(slice(0, min_len))

        slicing = tuple(slice_lst)

        if can_trim:
            return body_nd[slicing]

        out = chunk_spec.prototype.nd_buffer.create(
            shape=chunk_spec.shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
            fill_value=chunk_spec.fill_value,
        )
        out[slicing] = body_nd[slicing]
        return out

    # async def _encode_single(
    #     self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    # ) -> Buffer | None:
    #     header = N5BlockHeader(N5Mode.DEFAULT, chunk_spec.shape)
    #     for c in self.codecs:
    #         chunk_data = c._encode_single(chunk_data, chunk_spec)  # type:ignore
    #         chunk_spec = c.resolve_metadata(chunk_spec)

    #     buf: Buffer = chunk_data  # type: ignore

    #     bio = BytesIO()
    #     bio.write(header.to_bytes())
    #     # TODO: avoid this copy?
    #     bio.write(buf.as_buffer_like())
    #     return Buffer.from_bytes(bio.getbuffer())

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
        return {
            "name": N5_DEFAULT_NAME,
            "configuration": {"codecs": [c.to_dict() for c in self.codecs]},
        }
