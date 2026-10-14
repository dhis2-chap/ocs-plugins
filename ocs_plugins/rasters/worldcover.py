"""ESA WorldCover 10 m land cover (2021, v200) from the public AWS Open Data bucket.

WorldCover classifies the land surface at 10 m into 11 classes, in 3x3 degree cloud-optimised
GeoTIFF tiles on a 1/12000 degree EPSG:4326 grid with pixel edges on whole degrees. Ocean tiles
do not exist; the bucket's tile grid says which do. No account is needed.

At 10 m a country is large: Rwanda is about 575 million cells. The tiles are therefore opened
lazily with dask and only the windows inside the extent are read, chunk by chunk, as the store is
written, so memory stays at a few chunks. The classes are kept as ``uint8`` with 0 as the fill
value, as published, rather than widened to floats.
"""

from __future__ import annotations

import json
import math
import urllib.request
from functools import lru_cache
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
_YEAR, _VERSION = "2021", "v200"
_TILE_DEGREES = 3
_RES = 1 / 12000
_NO_DATA = 0
_CHUNK = 4096


@lru_cache(maxsize=1)
def _available_tiles() -> frozenset[str]:
    """Names (lower-left corner, e.g. ``S03E027``) of the tiles the bucket holds."""
    with urllib.request.urlopen(f"{_BUCKET}/{_VERSION}/{_YEAR}/esa_worldcover_grid.geojson", timeout=60) as response:
        grid = json.load(response)
    return frozenset(feature["properties"]["ll_tile"] for feature in grid["features"])


def _tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"{ns}{ew}"


def _tile_path(name: str) -> str:
    return f"/vsicurl/{_BUCKET}/{_VERSION}/{_YEAR}/map/ESA_WorldCover_10m_{_YEAR}_{_VERSION}_{name}_Map.tif"


def _tiles_for(bbox: list[float]) -> list[str]:
    west, south, east, north = bbox
    step = _TILE_DEGREES
    lats = range(math.floor(south / step) * step, math.ceil(north / step) * step, step)
    lons = range(math.floor(west / step) * step, math.ceil(east / step) * step, step)
    available = _available_tiles()
    return [name for name in (_tile_name(lat, lon) for lat in lats for lon in lons) if name in available]


def _grid_axes(bbox: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of the output: ``bbox`` widened to whole pixels of the product grid."""
    first_col, last_col = math.floor(bbox[0] / _RES), math.ceil(bbox[2] / _RES)
    first_row, last_row = math.floor(bbox[1] / _RES), math.ceil(bbox[3] / _RES)
    x = (np.arange(first_col, last_col) + 0.5) * _RES
    y = (np.arange(last_row - 1, first_row - 1, -1) + 0.5) * _RES  # north first
    return x, y


class WorldCoverPlugin(BaseDatasetPlugin):
    """ESA WorldCover 2021 land cover classes (10 tree cover ... 100 moss and lichen; 0 is no data)."""

    async def periods(self, start: str, end: str) -> list[str]:
        return [_YEAR] if int(start[:4]) <= int(_YEAR) <= int(end[:4]) else []

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rioxarray

        tiles = _tiles_for(bbox)
        if not tiles:
            raise RuntimeError(f"No ESA WorldCover tiles intersect {bbox}")
        x, y = _grid_axes(bbox)

        parts = []
        for name in tiles:
            tile = cast(
                xr.DataArray,
                rioxarray.open_rasterio(_tile_path(name), masked=False, chunks={"x": _CHUNK, "y": _CHUNK}, lock=False),
            ).squeeze("band", drop=True)
            # Snap the tile's coordinates onto the exact grid, so tiles meet without gaps or overlaps.
            tile = tile.assign_coords(
                x=(np.round(tile.x.values / _RES - 0.5) + 0.5) * _RES,
                y=(np.round(tile.y.values / _RES - 0.5) + 0.5) * _RES,
            )
            inside = tile.sel(x=slice(x[0] - _RES / 2, x[-1] + _RES / 2), y=slice(y[0] + _RES / 2, y[-1] - _RES / 2))
            if inside.size:
                parts.append(inside.reindex(x=x, y=y, method="nearest", tolerance=_RES / 4, fill_value=_NO_DATA))

        if not parts:
            raise RuntimeError(f"No ESA WorldCover data within {bbox}")
        # Tiles do not overlap, and a cell outside a tile is 0, so the largest value is the tile's own.
        classes = parts[0] if len(parts) == 1 else xr.concat(parts, dim="tile").max("tile")
        classes = classes.astype("uint8").drop_vars([c for c in classes.coords if c not in ("x", "y")])
        classes.attrs = {"units": "1", "flag_meanings": "ESA WorldCover class codes, see the template description"}
        classes.encoding = {"_FillValue": np.uint8(_NO_DATA)}
        ds = classes.to_dataset(name="landcover").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01-01", "ns")]}))
