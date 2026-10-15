"""MODIS land cover (MCD12Q1 v061, IGBP classes, 500 m, yearly) from NASA LP DAAC.

MCD12Q1 classifies each 500 m cell of the MODIS sinusoidal tile grid once a year, from 2001,
using Terra and Aqua. This plugin stores ``LC_Type1``, the 17-class IGBP scheme, as published.

Granules are found through CMR and downloaded with the Earthdata Login credentials (see
``earthdata``). They are HDF4 files, which GDAL builds from PyPI cannot read, so they are read
with ``pyhdf``. Each tile is placed on the sinusoidal grid from its h/v index and reprojected to
EPSG:4326 at ``resolution`` degrees (default 1/240, about 500 m) with nearest-neighbour
resampling, since class codes cannot be averaged. All tiles go onto one output grid, so an
extent spanning several tiles is complete.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

from ocs_plugins.rasters.earthdata import cmr_granules, data_links, download, earthdata_session

_SHORT_NAME = "MCD12Q1"
_VERSION = "061"
_FIELD = "LC_Type1"
# A small land area every year's product covers, to find the latest published year.
_PROBE_BBOX = [29.5, -2.0, 29.6, -1.9]
_FILL = 255
_FIRST_YEAR = 2001

# The MODIS sinusoidal tile grid.
_SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
_TILE_SIZE = 1111950.5196666666
_GRID_WEST = -20015109.354
_GRID_NORTH = 10007554.677
_TILE = re.compile(r"\.h(\d{2})v(\d{2})\.")


def _tile_index(title: str) -> tuple[int, int]:
    match = _TILE.search(title)
    if match is None:
        raise RuntimeError(f"Cannot read the sinusoidal tile from granule {title!r}")
    return int(match.group(1)), int(match.group(2))


def _latest_year() -> int | None:
    """The latest year with a published MCD12Q1 v061 map, from CMR."""
    from datetime import UTC, datetime

    this_year = datetime.now(UTC).year
    granules = cmr_granules(
        _SHORT_NAME,
        (f"{this_year - 5}-01-01T00:00:00Z", f"{this_year}-12-31T23:59:59Z"),
        bbox=_PROBE_BBOX,
        version=_VERSION,
    )
    return max((int(granule["time_start"][:4]) for granule in granules), default=None)


def _read_classes(path: Path) -> np.ndarray:
    from pyhdf.SD import SD, SDC

    hdf = SD(str(path), SDC.READ)
    try:
        return np.asarray(hdf.select(_FIELD).get())
    finally:
        hdf.end()


def _target_grid(bbox: list[float], resolution: float) -> tuple[Any, int, int]:
    """The EPSG:4326 output grid: ``bbox`` widened to multiples of ``resolution``."""
    from rasterio.transform import from_origin

    west = math.floor(bbox[0] / resolution) * resolution
    south = math.floor(bbox[1] / resolution) * resolution
    east = math.ceil(bbox[2] / resolution) * resolution
    north = math.ceil(bbox[3] / resolution) * resolution
    return (
        from_origin(west, north, resolution, resolution),
        round((east - west) / resolution),
        round((north - south) / resolution),
    )


class MODISLandCoverPlugin(BaseDatasetPlugin):
    """Yearly IGBP land cover class codes (1-17) from MCD12Q1.

    Template params:
        resolution: output grid spacing in degrees (default 1/240).
    """

    def __init__(self, resolution: float = 1 / 240, **_: Any) -> None:
        self.resolution = float(resolution)

    async def periods(self, start: str, end: str) -> list[str]:
        import asyncio

        # Each year's map is published well after the year ends, so stop at the latest one out.
        latest = await asyncio.to_thread(_latest_year)
        if latest is None:
            return []
        first, last = max(int(start[:4]), _FIRST_YEAR), min(int(end[:4]), latest)
        return [str(year) for year in range(first, last + 1)]

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject

        year = period_id[:4]
        granules = cmr_granules(
            _SHORT_NAME, (f"{year}-01-01T00:00:00Z", f"{year}-01-01T23:59:59Z"), bbox=bbox, version=_VERSION
        )
        if not granules:
            raise RuntimeError(f"No MCD12Q1 land cover for {year} over {bbox}")

        transform, width, height = _target_grid(bbox, self.resolution)
        classes = np.full((height, width), _FILL, dtype="uint8")
        session = earthdata_session()
        with tempfile.TemporaryDirectory() as tmp:
            for granule in granules:
                links = data_links(granule, ".hdf")
                if not links:
                    raise RuntimeError(f"Granule {granule['title']} has no HDF data link")
                path = download(links[0], Path(tmp) / Path(links[0]).name, session=session)
                tile = _read_classes(path)
                path.unlink()
                h, v = _tile_index(granule["title"])
                rows, cols = tile.shape
                # Each tile fills only its own part of the output; the rest keeps what is there.
                reproject(
                    tile,
                    classes,
                    src_transform=from_origin(
                        _GRID_WEST + h * _TILE_SIZE, _GRID_NORTH - v * _TILE_SIZE, _TILE_SIZE / cols, _TILE_SIZE / rows
                    ),
                    src_crs=CRS.from_proj4(_SINUSOIDAL),
                    src_nodata=_FILL,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=_FILL,
                    resampling=Resampling.nearest,
                    init_dest_nodata=False,
                )

        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        values = np.where(classes == _FILL, np.nan, classes.astype("float32")).astype("float32")
        data = xr.DataArray(
            values,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "1", "flag_meanings": "IGBP class codes, see the template description"},
        )
        ds = data.to_dataset(name="land_cover").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{year}-01-01", "ns")]}))
