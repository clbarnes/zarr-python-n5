import json

from zarr.storage import MemoryStore

from zarr.codecs import BytesCodec
from zarr.core.metadata.v3 import ArrayV3Metadata

from zarr.buffer import default_buffer_prototype
from zarr_n5 import N5WrapperStore, N5DefaultCodec

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(["dtype"], [("uint8",), ("float32",)])
async def test_bytes_endian(dtype: str):
    proto = default_buffer_prototype()
    attributes = {
        "blockSize": [256, 128],
        "compression": {"type": "raw"},
        "dataType": dtype,
        "dimensions": [256, 128],
    }
    s = json.dumps(attributes)
    b = s.encode()
    buffer = proto.buffer.from_bytes(b)
    data = {"attributes.json": buffer}
    inner = MemoryStore(data)
    store = N5WrapperStore(inner)

    res = await store.get("zarr.json", proto)
    assert res is not None
    b2 = res.to_bytes()
    s2 = b2.decode()
    d2 = json.loads(s2)
    meta = ArrayV3Metadata.from_dict(d2)
    n5codec = meta.codecs[0]
    assert isinstance(n5codec, N5DefaultCodec)
    bts = n5codec.codecs[-1]
    assert isinstance(bts, BytesCodec)
    assert bts.endian is None or str(bts.endian.value) == "big"
