from typing import Final
from collections.abc import Iterable
import json

from zarr.abc.store import (
    Store,
    ByteRequest,
)
from zarr.storage import WrapperStore
from zarr.core.group import GroupMetadata
from zarr.core.buffer import Buffer, BufferPrototype

from ..util import slice_buf, is_metadata


def make_implicit_group_bytes() -> Buffer:
    g = GroupMetadata()
    g.attributes["_implicit"] = True

    return Buffer.from_bytes(json.dumps(g.to_dict()).encode())


IMPLICIT_GROUP_BUFFER: Final[Buffer] = make_implicit_group_bytes()


class ImplicitGroupWrapperStore(WrapperStore):
    """A store which supplies empty group metadata documents if they do not exist.

    Used to replicate N5's behaviour where any directory (or prefix) is a valid group,
    even when no metadata document exists.
    Wrap over an `N5WrapperStore`.
    """

    _store: Store

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        res = await self._store.get(key, prototype, byte_range)
        if res is not None or not is_metadata(key):
            return res

        b = slice_buf(IMPLICIT_GROUP_BUFFER.as_buffer_like(), byte_range)
        return Buffer.from_bytes(b)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        key_ranges = list(key_ranges)
        reses = await super().get_partial_values(prototype, key_ranges)
        out = []
        for (key, byte_range), res in zip(key_ranges, reses):
            if res is not None or not is_metadata(key):
                out.append(res)
                continue

            b = Buffer.from_bytes(
                slice_buf(IMPLICIT_GROUP_BUFFER.as_buffer_like(), byte_range)
            )
            out.append(b)

        return out

    async def exists(self, key: str) -> bool:
        if is_metadata(key):
            return True
        return await super().exists(key)
