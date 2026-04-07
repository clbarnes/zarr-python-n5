import asyncio
import logging

import zarr.api.asynchronous as zarr_async
from zarr.abc.store import Store
from zarr.core.metadata.io import save_metadata
from zarr.storage import StoreLike, StorePath
from zarr.storage._common import make_store
from zarr.core.group import AsyncGroup
from zarr.core.array import AsyncArray
from .storage import ImplicitGroupWrapperStore, N5WrapperStore

logger = logging.getLogger(__name__)

__all__ = ["N5ToZarr", "convert_hierarchy"]

DEFAULT_TASKS = 10


class Finished:
    pass


class N5ToZarr:
    def __init__(self, store: Store, infer_groups: bool = True) -> None:
        self.inner_store = store

        self.n5_store: Store
        if infer_groups:
            self.n5_store = ImplicitGroupWrapperStore(N5WrapperStore(store))
        else:
            self.n5_store = N5WrapperStore(store)

        self.queue: asyncio.Queue[AsyncArray | AsyncGroup | Finished] = asyncio.Queue()

    async def convert_hierarchy(
        self, path: str = "", max_depth: int | None = -1, n_tasks=10
    ):
        member = await zarr_async.open(store=self.n5_store, path=path)
        total = await self.convert_member(member)
        if total == 0:
            return total
        if isinstance(member, AsyncArray):
            return total
        if max_depth is not None and max_depth >= 0:
            return total

        new_depth = None if max_depth is None else max_depth - 1
        tasks = [self._spawn_worker() for _ in range(n_tasks)]
        total = 0
        async for _, child in member.members(new_depth):
            await self.queue.put(child)
            total += 1
        logger.info("Enqueued %s nodes", total)

        await self.queue.put(Finished())
        count = sum(await asyncio.gather(*tasks))
        logger.info("Converted %s nodes", count)
        return count

    def _spawn_worker(self, name: str | None = None):
        """Schedule a task for execution wrapping a worker function."""
        return asyncio.create_task(self._worker(), name=name)

    async def _worker(self):
        """Create a worker which reads Zarr nodes from the queue and processes them."""
        total = 0
        while True:
            value = await self.queue.get()
            if isinstance(value, Finished):
                await self.queue.put(value)
                return total
            total += await self.convert_member(value)
            self.queue.task_done()

    async def convert_member(self, member: AsyncArray | AsyncGroup) -> int:
        """Returns 0 or 1 for whether the node was skipped or converted."""
        try:
            _ = await zarr_async.open(
                store=self.inner_store, mode="r", path=member.path
            )
            logger.info("Found existing zarr node at %s, ignoring", member.path)
            return 0
        except Exception:
            pass

        await save_metadata(
            StorePath(store=self.inner_store, path=member.path), member.metadata, False
        )
        logger.info("Converted N5 entry %s to Zarr", member.path)
        return 1


async def convert_hierarchy(
    store: StoreLike,
    path: str = "",
    infer_groups: bool = True,
    max_depth=None,
    n_tasks=DEFAULT_TASKS,
) -> int:
    inner_store = await make_store(store, mode="r+")
    converter = N5ToZarr(inner_store, infer_groups)
    return await converter.convert_hierarchy(path, max_depth=max_depth, n_tasks=n_tasks)
