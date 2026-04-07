# zarr-python-n5

[N5](https://github.com/saalfeldlab/n5) utilities for [zarr-python](https://github.com/zarr-developers/zarr-python).

## Codecs

### N5 Default Codec

[As described here](https://github.com/zarr-developers/zarr-extensions/pull/49).

Only whole-chunk reading is supported.

#### N5 Compressor support

| N5 compressor | Supported | Zarr bytes-to-bytes codec | Notes |
| ------------- | --------- | ------------------------- | ----- |
| `raw` | yes | n/a | Equivalent to omitted bytes-to-bytes codec |
| `blosc` | yes | `blosc` | |
| `gzip` | yes | `gzip` | |
| `zstd` | yes | `zstd` | |
| `lz4` | no | | [Incompatible codecs](https://github.com/zarr-developers/numcodecs/issues/175) |
| `xz` | no | | No equivalent Zarr codec |
| `jpeg` | no | | Needs [N5 documentation](https://github.com/saalfeldlab/n5-jpeg/issues/1), [Zarr codec](https://github.com/zarr-developers/zarr-extensions/issues/15) |
| `bzip2` | no | | No equivalent Zarr codec |

## Stores

`N5WrapperStore` allows reading N5 data with DEFAULT-mode blocks through any Zarr store by converting metadata on the fly.
By default, this does not replicate the N5 behaviour of inferring an empty group where a metadata document does not exist.
To achieve this, wrap it in the provided `ImplicitGroupWrapperStore`.

## Contributing

Use [`uv`](https://docs.astral.sh/uv/) for project management.

Use [`just`](https://github.com/casey/just) for common development tasks.
