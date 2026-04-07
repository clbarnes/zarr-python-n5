"""
Module containing `N5WrapperStore`,
for silently converting N5 nodes to Zarr nodes.
"""

from collections import defaultdict
from collections.abc import AsyncIterator, Iterable
from zarr.storage import WrapperStore
from zarr.abc.store import (
    Store,
    ByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
import json
import asyncio

from ..constants import N5_METADATA_KEY, ZARR_V3_METADATA_KEY
from ..metadata import N5GroupMetadata, N5ArrayMetadata
from ..util import slice_buf, is_zarr3_metadata, N5Mode


class N5WrapperStore[T: Store](WrapperStore):
    """A read-only store for opening N5 hierarchies.

    Requests for Zarr metadata documents are redirected to N5 attributes,
    and Zarr metadata calculated on the fly.

    Note that N5 attributes can be omitted in groups.
    You may want to wrap this in an `ImplicitGroupWrapperStore` to replicate that behaviour.

    Only compatible with DEFAULT-mode N5 arrays.
    """

    _store: T

    def intercept_metadata(self, key: str) -> None | str:
        """If the given key is for Zarr v3 metadata, return the key for N5 metadata in the equivalent node.

        Otherwise, return None.
        """
        if "/" in key:
            pref, fname = key.rsplit("/", 1)
        else:
            pref = None
            fname = key

        if fname != ZARR_V3_METADATA_KEY:
            return None

        if pref is None:
            k2 = N5_METADATA_KEY
        else:
            k2 = f"{pref}/{N5_METADATA_KEY}"

        return k2

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        k2 = self.intercept_metadata(key)
        if k2 is None:
            return await self._store.get(key, prototype, byte_range)

        b = await self._store.get(k2, prototype)

        if b is None:
            return None

        d = json.loads(b.to_bytes())
        n5_meta = N5GroupMetadata.from_jso(d)
        try:
            n5_meta = N5ArrayMetadata.from_group(n5_meta)
            out_d = n5_meta.to_zarr(N5Mode.DEFAULT)
        except KeyError:
            out_d = n5_meta.to_zarr()

        b2 = json.dumps(out_d.to_dict()).encode()
        b2 = slice_buf(b2, byte_range)

        return prototype.buffer.from_bytes(b2)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:

        # Split the key ranges into metadata requests and other (chunk) requests.
        # We always need to read the whole N5 metadata file
        # to convert it into Zarr v3 metadata before slicing it,
        # so this prevents reading it multiple times.
        meta_reqs: defaultdict[str, list[tuple[int, ByteRequest | None]]] = defaultdict(
            list
        )
        other_reqs: list[tuple[int, tuple[str, ByteRequest | None]]] = []
        count = 0
        for idx, (key, byte_range) in enumerate(key_ranges):
            if is_zarr3_metadata(key):
                meta_reqs[key].append((idx, byte_range))
            else:
                other_reqs.append((idx, (key, byte_range)))
            count += 1

        other_reqs_fut = self._store.get_partial_values(
            prototype, (tup[1] for tup in other_reqs)
        )
        meta_req_list = list(meta_reqs.items())
        meta_reqs_fut = asyncio.gather(
            *(self.get(k, prototype) for k, _ in meta_req_list)
        )
        # Gather all requests to run concurrently
        other_res, meta_res = await asyncio.gather(other_reqs_fut, meta_reqs_fut)
        out: list[None | Buffer] = [None for _ in range(count)]

        # Slice and insert the metadata responses into the pre-allocated output list
        for res, (_, meta_req) in zip(meta_res, meta_req_list):
            if res is None:
                continue
            blike = res.as_buffer_like()
            for idx, byte_range in meta_req:
                out[idx] = Buffer.from_bytes(slice_buf(blike, byte_range))

        # Insert the non-metadata responses into the output;
        # these are already sliced by the underlying store.
        for res, (idx, _) in zip(other_res, other_reqs):
            out[idx] = res

        return out

    async def exists(self, key: str) -> bool:
        k2 = self.intercept_metadata(key)
        return await self._store.exists(k2 or key)

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        return self._store.supports_listing

    def list(self) -> AsyncIterator[str]:
        return self._store.list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._store.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._store.list_dir(prefix)
