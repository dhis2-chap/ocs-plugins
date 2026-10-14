"""Copernicus DEM GLO-30 and GLO-90 from the public AWS Open Data buckets (no account needed).

The 1x1 degree COG tiles are mosaicked and clipped to the extent on their native grid: 1
arc-second (about 30 m) for GLO-30, 3 arc-seconds (about 90 m) for GLO-90, in EPSG:4326. Ocean
areas have no tiles, so each bucket's `tileList.txt` decides which tiles to read, and cells
without a tile are NaN. Only the windows inside the extent are read, over HTTP range requests.

Tiles above 50 degrees latitude are stored with wider pixels in longitude; they are resampled
bilinearly onto the common grid. The static dataset is represented as a single yearly period;
2021 is the year the release was published.
"""

import math
import urllib.request
from functools import lru_cache
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

# Resolution in metres -> (bucket, tile name code, pixel size in degrees).
_PRODUCTS = {
    30: ("https://copernicus-dem-30m.s3.amazonaws.com", "10", 1 / 3600),
    90: ("https://copernicus-dem-90m.s3.amazonaws.com", "30", 1 / 1200),
}
_PERIOD = "2021"


@lru_cache(maxsize=len(_PRODUCTS))
def _available_tiles(bucket: str) -> frozenset[str]:
    """Names of the tiles a bucket holds; ocean tiles are absent."""
    with urllib.request.urlopen(f"{bucket}/tileList.txt", timeout=60) as response:
        return frozenset(line.strip() for line in response.read().decode().splitlines() if line.strip())


def _tile_name(code: str, lat: int, lon: int) -> str:
    """The tile whose south-west corner is at ``lat``, ``lon`` (whole degrees)."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_{code}_{ns}_00_{ew}_00_DEM"


def _tile_path(bucket: str, name: str) -> str:
    return f"/vsicurl/{bucket}/{name}/{name}.tif"


def _snap_outward(bbox: list[float], res: float) -> tuple[float, float, float, float]:
    """Widen ``bbox`` to the tiles' pixel edges, which sit half a pixel off whole degrees."""
    half = res / 2
    west, south, east, north = bbox
    return (
        math.floor((west + half) / res) * res - half,
        math.floor((south + half) / res) * res - half,
        math.ceil((east + half) / res) * res - half,
        math.ceil((north + half) / res) * res - half,
    )


class CopernicusDEMPlugin(BaseDatasetPlugin):
    """Streaming plugin for Copernicus DEM elevation.

    Template params:
        resolution: 30 (GLO-30) or 90 (GLO-90, the default).
    """

    def __init__(self, resolution: int = 90, **_: Any) -> None:
        if int(resolution) not in _PRODUCTS:
            raise ValueError(f"{type(self).__name__}: resolution must be one of {list(_PRODUCTS)}, got {resolution!r}")
        self.bucket, self.code, self.res = _PRODUCTS[int(resolution)]

    async def periods(self, start: str, end: str) -> list[str]:
        return [_PERIOD] if int(start[:4]) <= int(_PERIOD) <= int(end[:4]) else []

    def _tiles_for(self, bbox: list[float]) -> list[str]:
        west, south, east, north = bbox
        names = (
            _tile_name(self.code, lat, lon)
            for lat in range(math.floor(south), math.ceil(north))
            for lon in range(math.floor(west), math.ceil(east))
        )
        available = _available_tiles(self.bucket)
        return [name for name in names if name in available]

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rasterio
        import rasterio.merge
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.enums import Resampling

        tiles = self._tiles_for(bbox)
        if not tiles:
            raise RuntimeError(f"No Copernicus DEM tiles intersect {bbox}")
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            sources = [rasterio.open(_tile_path(self.bucket, name)) for name in tiles]
            try:
                mosaic, transform = rasterio.merge.merge(
                    sources,
                    bounds=_snap_outward(bbox, self.res),
                    res=(self.res, self.res),
                    nodata=np.nan,
                    dtype="float32",
                    resampling=Resampling.bilinear,
                )
            finally:
                for source in sources:
                    source.close()

        height, width = mosaic.shape[1:]
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        elevation = xr.DataArray(mosaic[0], dims=("y", "x"), coords={"y": y, "x": x}, attrs={"units": "m"})
        ds = elevation.to_dataset(name="elevation").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(period_id, "ns")]}))
