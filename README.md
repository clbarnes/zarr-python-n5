# zarr-python-n5

[N5](https://github.com/saalfeldlab/n5) utilities for [zarr-python](https://github.com/zarr-developers/zarr-python).

## Codecs

### N5 Default Codec

[As described here](https://github.com/zarr-developers/zarr-extensions/pull/49).

## Stores

`N5WrapperStore` allows reading N5 data with DEFAULT-mode blocks through any Zarr store by converting metadata on the fly.
By default, this does not replicate the N5 behaviour of inferring an empty group where a metadata document does not exist.
To achieve this, wrap it in the provided `ImplicitGroupWrapperStore`.
