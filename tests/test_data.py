import pytest
import zarr
from zarr.abc.store import Store
import numpy as np
from numpy.testing import assert_allclose


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
def test_data(name: str, data_store: Store, raw_data: np.ndarray):
    arr = zarr.open_array(data_store, path=f"{name}.n5")
    read = np.array(arr)
    assert_allclose(read, raw_data)
