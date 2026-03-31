from zarr.registry import register_codec

from .codec.default import N5DefaultCodec
from .storage.n5 import N5WrapperStore
from .storage.implicit import ImplicitGroupWrapperStore

__all__ = ["N5WrapperStore", "ImplicitGroupWrapperStore", "N5DefaultCodec"]

register_codec("n5_default", N5DefaultCodec, qualname="zarr_n5.N5DefaultCodec")
