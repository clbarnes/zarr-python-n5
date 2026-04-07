from pathlib import Path
import shutil
import asyncio

import pytest
import zarr
from zarr_n5.convert import convert_hierarchy


@pytest.mark.parametrize(
    ["name"],
    [
        ("blosc",),
        ("even_chunk",),
        ("gzip",),
        ("single_chunk",),
        ("uneven_chunk_padded",),
        ("uneven_chunk_truncated",),
        ("zstd",),
    ],
)
def test_convert(data_dir: Path, tmpdir, name: str):
    dname = f"{name}.n5"
    src = data_dir / f"{name}.n5"
    tmp_path = Path(tmpdir)
    tgt = tmp_path / dname
    shutil.copytree(src, tgt)
    asyncio.run(convert_hierarchy(tmp_path, dname))
    _ = zarr.open(tgt)
