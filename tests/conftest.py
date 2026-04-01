from pathlib import Path
import pytest
import numpy as np
from zarr.abc.store import Store
from zarr.storage import LocalStore

from zarr_n5 import N5WrapperStore, ImplicitGroupWrapperStore


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def raw_data(data_dir) -> np.ndarray:
    return np.load(data_dir / "raw.npy")


@pytest.fixture
def data_store(data_dir) -> Store:
    local = LocalStore(data_dir)
    n5: N5WrapperStore = N5WrapperStore(local)
    implicit_group: ImplicitGroupWrapperStore = ImplicitGroupWrapperStore(n5)
    return implicit_group
