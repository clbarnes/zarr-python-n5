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
        self,
        path: str = "",
        max_depth: int = -1,
        n_tasks=10,
        overwrite_zarr_json: bool = False,
    ):
        """Convert an N5 hierarchy into zarr.

        No existing objects are altered and no chunk data is copied;
        a zarr.json is simply added to every N5 node.

        Parameters
        ----------
        path
            The path of the node to start at.
        max_depth
            <0: find all descendant nodes (default).
            0: only convert the root.
            n: convert descendants down to `n` levels (immediate children are 1, grandchildren are 2 etc.).
        n_tasks
            Number of async tasks started, by default 10
        overwrite_zarr_json
            Whether to overwrite existing zarr.json objects, by default False.
            True saves having to check for (valid) zarr.json objects and so may be faster.

        Returns
        -------
        count
            Total number of nodes converted.

        Raises
        ------
        ValueError
            Invalid number of tasks (must be >= 1).
        """
        member = await zarr_async.open(store=self.n5_store, path=path)

        if n_tasks < 1:
            raise ValueError(f"Must use at least 1 task, got {n_tasks}")

        tasks = []
        await self.queue.put(member)
        tasks.append(self._spawn_worker(overwrite_zarr_json))

        enqueued = 1
        if isinstance(member, AsyncGroup) and max_depth != 0:
            if max_depth < 0:
                md = None
            else:
                md = max_depth - 1

            async for _, child in member.members(md):
                await self.queue.put(child)
                if len(tasks) < n_tasks:
                    tasks.append(self._spawn_worker(overwrite_zarr_json))

                enqueued += 1

        logger.info("Enqueued %s nodes", enqueued)

        await self.queue.put(Finished())
        count = sum(await asyncio.gather(*tasks))
        logger.info("Converted %s nodes", count)
        return count

    def _spawn_worker(self, overwrite_zarr_json: bool, name: str | None = None):
        """Schedule a task for execution wrapping a worker function."""
        return asyncio.create_task(self._worker(overwrite_zarr_json), name=name)

    async def _worker(self, overwrite_zarr_json: bool):
        """Create a worker which reads Zarr nodes from the queue and processes them."""
        total = 0
        while True:
            value = await self.queue.get()
            if isinstance(value, Finished):
                await self.queue.put(value)
                return total
            total += await self.convert_member(value, overwrite_zarr_json)
            self.queue.task_done()

    async def convert_member(
        self, member: AsyncArray | AsyncGroup, overwrite_zarr_json: bool = False
    ) -> int:
        """Returns 0 or 1 for whether the node was skipped or converted."""
        if not overwrite_zarr_json:
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
    max_depth: int = -1,
    n_tasks: int = DEFAULT_TASKS,
    overwrite_zarr_json: bool = False,
) -> int:
    inner_store = await make_store(store, mode="r+")
    converter = N5ToZarr(inner_store, infer_groups)
    return await converter.convert_hierarchy(
        path,
        max_depth=max_depth,
        n_tasks=n_tasks,
        overwrite_zarr_json=overwrite_zarr_json,
    )
