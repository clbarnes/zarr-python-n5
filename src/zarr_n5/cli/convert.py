"""n5tozarr command line interface."""

from argparse import ArgumentParser
import asyncio

from ..convert import convert_hierarchy, DEFAULT_TASKS


def main(raw_args=None):
    """n5tozarr main function."""
    parser = ArgumentParser("n5tozarr")
    parser.add_argument("url", help="URL to Zarr store, using fsspec format")
    parser.add_argument(
        "path", help="paths within the Zarr store to process", nargs="?"
    )
    parser.add_argument(
        "-t", "--tasks", type=int, default=DEFAULT_TASKS, help="asynchronous task count"
    )
    parser.add_argument(
        "-d", "--max-depth", type=int, help="how far to recurse; default no maximum"
    )
    parser.add_argument(
        "-I",
        "--no-infer-groups",
        action="store_true",
        help="do not infer N5 groups from empty directories/ prefixes",
    )

    args = parser.parse_args(raw_args)
    fut = convert_hierarchy(
        args.url, args.path or "", not args.no_infer_groups, args.max_depth, args.tasks
    )
    asyncio.run(fut)
