"""
Module containing `ImplicitGroupWrapperStore`,
for inferring groups with missing metadata.
"""

from typing import Final
from collections.abc import Iterable
import json

from zarr.abc.store import (
    Store,
    ByteRequest,
)
from zarr.storage import WrapperStore
from zarr.core.group import GroupMetadata
from zarr.core.buffer import BufferPrototype, Buffer

from ..util import slice_buf, is_zarr3_metadata

__all__ = ["ImplicitGroupWrapperStore"]


def make_implicit_group_bytes() -> bytes:
    g = GroupMetadata()
    g.attributes["_implicit"] = True

    return json.dumps(g.to_dict()).encode()


IMPLICIT_GROUP_BYTES: Final[bytes] = make_implicit_group_bytes()


class ImplicitGroupWrapperStore[T: Store](WrapperStore):
    """A store which supplies empty group metadata documents if they do not exist.

    Used to replicate N5's behaviour where any directory (or prefix) is a valid group,
    even when no metadata document exists.
    Wrap over an `N5WrapperStore`.

    Inferred group metadata's attributes will contain the key/value `"_implicit": true`.
    """

    _store: T

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        res = await self._store.get(key, prototype, byte_range)
        if res is not None or not is_zarr3_metadata(key):
            return res

        b = slice_buf(IMPLICIT_GROUP_BYTES, byte_range)
        return prototype.buffer.from_bytes(b)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        key_ranges = list(key_ranges)
        reses = await super().get_partial_values(prototype, key_ranges)
        out = []
        for (key, byte_range), res in zip(key_ranges, reses):
            if res is None and is_zarr3_metadata(key):
                res = prototype.buffer.from_bytes(
                    slice_buf(IMPLICIT_GROUP_BYTES, byte_range)
                )
            out.append(res)

        return out

    async def exists(self, key: str) -> bool:
        if is_zarr3_metadata(key):
            return True
        return await super().exists(key)
