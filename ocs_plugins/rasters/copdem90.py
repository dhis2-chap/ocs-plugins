"""Copernicus DEM GLO-90 from the public AWS Open Data bucket (no account needed).

The 1x1 degree COG tiles are mosaicked and clipped to the extent on their native
3 arc-second (about 90 m) EPSG:4326 grid. Ocean areas have no tiles, so the bucket's
`tileList.txt` decides which tiles to read, and cells without a tile are NaN.

Tiles above 50 degrees latitude are stored with wider pixels in longitude; they are
resampled bilinearly onto the common 3 arc-second grid. Like the GLO-30 plugin, the static
dataset is represented as a single yearly period; 2021 is the year the release was published.
"""

import math
import urllib.request
from functools import lru_cache
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

_BUCKET_URL = "https://copernicus-dem-90m.s3.amazonaws.com"
_RESOLUTION = 1 / 1200
_PERIOD = "2021"


@lru_cache(maxsize=1)
def _available_tiles() -> frozenset[str]:
    """Names of the tiles the bucket holds; ocean tiles are absent."""
    with urllib.request.urlopen(f"{_BUCKET_URL}/tileList.txt", timeout=60) as response:
        return frozenset(line.strip() for line in response.read().decode().splitlines() if line.strip())


def _tile_name(lat: int, lon: int) -> str:
    """The tile whose south-west corner is at ``lat``, ``lon`` (whole degrees)."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def _tiles_for(bbox: list[float]) -> list[str]:
    west, south, east, north = bbox
    names = (
        _tile_name(lat, lon)
        for lat in range(math.floor(south), math.ceil(north))
        for lon in range(math.floor(west), math.ceil(east))
    )
    available = _available_tiles()
    return [name for name in names if name in available]


def _snap_outward(bbox: list[float]) -> tuple[float, float, float, float]:
    """Widen ``bbox`` to the tiles' pixel edges, which sit half a pixel off whole degrees."""
    half = _RESOLUTION / 2
    west, south, east, north = bbox
    return (
        math.floor((west + half) / _RESOLUTION) * _RESOLUTION - half,
        math.floor((south + half) / _RESOLUTION) * _RESOLUTION - half,
        math.ceil((east + half) / _RESOLUTION) * _RESOLUTION - half,
        math.ceil((north + half) / _RESOLUTION) * _RESOLUTION - half,
    )


class CopDEM90Plugin(BaseDatasetPlugin):
    """Streaming plugin for Copernicus DEM GLO-90 elevation."""

    async def periods(self, start: str, end: str) -> list[str]:
        return [_PERIOD] if int(start[:4]) <= int(_PERIOD) <= int(end[:4]) else []

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rasterio
        import rasterio.merge
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.enums import Resampling

        tiles = _tiles_for(bbox)
        if not tiles:
            raise RuntimeError(f"No Copernicus DEM GLO-90 tiles intersect {bbox}")
        bounds = _snap_outward(bbox)
        # Only the windows inside `bounds` are read from each COG, over HTTP range requests.
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            sources = [rasterio.open(_tile_path(name)) for name in tiles]
            try:
                mosaic, transform = rasterio.merge.merge(
                    sources,
                    bounds=bounds,
                    res=(_RESOLUTION, _RESOLUTION),
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


def _tile_path(name: str) -> str:
    return f"/vsicurl/{_BUCKET_URL}/{name}/{name}.tif"
