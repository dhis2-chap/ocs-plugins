"""MODIS sinusoidal tiles from NASA LP DAAC: download, read and mosaic onto one EPSG:4326 grid.

Shared by the MODIS plugins. Granules are found through CMR and downloaded with the Earthdata
Login credentials (see ``earthdata``). They are HDF4 files, which GDAL builds from PyPI cannot
read, so they are read with ``pyhdf``. Each tile is placed on the MODIS sinusoidal grid from its
h/v index and reprojected into a single output grid with nearest-neighbour resampling, so an
extent spanning several tiles is complete and values are copied, never averaged.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ocs_plugins.rasters.earthdata import _CMR_GRANULES, cmr_granules, data_links, download, earthdata_session

SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
TILE_SIZE = 1111950.5196666666
GRID_WEST = -20015109.354
GRID_NORTH = 10007554.677
_TILE = re.compile(r"\.h(\d{2})v(\d{2})\.")


def tile_index(title: str) -> tuple[int, int]:
    """The (h, v) sinusoidal tile of a granule, from its title."""
    match = _TILE.search(title)
    if match is None:
        raise RuntimeError(f"Cannot read the sinusoidal tile from granule {title!r}")
    return int(match.group(1)), int(match.group(2))


def read_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The named scientific datasets of an HDF4 file, as stored."""
    from pyhdf.SD import SD, SDC

    hdf = SD(str(path), SDC.READ)
    try:
        return {name: np.asarray(hdf.select(name).get()) for name in names}
    finally:
        hdf.end()


def granules_for_day(short_name: str, version: str, day: str, bbox: list[float]) -> list[dict]:
    """The granules of ``short_name`` starting on ``day`` (``YYYY-MM-DD``) over ``bbox``."""
    found = cmr_granules(short_name, (f"{day}T00:00:00Z", f"{day}T23:59:59Z"), bbox=bbox, version=version)
    return [granule for granule in found if granule["time_start"][:10] == day]


def latest_granule_start(short_name: str, version: str, bbox: list[float]) -> str | None:
    """Start date (``YYYY-MM-DD``) of the newest granule over ``bbox``, from CMR, or None.

    One request, newest first, over the whole archive: a publication gap or the end of a
    collection must not hide the dates that are published.
    """
    import requests

    query = {
        "short_name": short_name,
        "version": version,
        "bounding_box": ",".join(str(float(v)) for v in bbox),
        "sort_key": "-start_date",
        "page_size": 1,
    }
    response = requests.get(_CMR_GRANULES, params=query, timeout=60)
    response.raise_for_status()
    entries = response.json()["feed"]["entry"]
    return entries[0]["time_start"][:10] if entries else None


def target_grid(bbox: list[float], resolution: float) -> tuple[Any, int, int]:
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


def mosaic(
    granules: list[dict],
    bbox: list[float],
    resolution: float,
    read: Callable[[Path], np.ndarray],
    fill: float,
    dtype: str,
) -> xr.DataArray:
    """Download each granule, ``read`` it into one 2-D array, and mosaic all onto one grid.

    ``read`` returns the tile's values with ``fill`` where there is no data; cells no tile covers
    are ``fill`` too.
    """
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    transform, width, height = target_grid(bbox, resolution)
    values = np.full((height, width), fill, dtype=dtype)
    session = earthdata_session()
    with tempfile.TemporaryDirectory() as tmp:
        for granule in granules:
            links = data_links(granule, ".hdf")
            if not links:
                raise RuntimeError(f"Granule {granule['title']} has no HDF data link")
            path = download(links[0], Path(tmp) / Path(links[0]).name, session=session)
            tile = read(path)
            path.unlink()
            h, v = tile_index(granule["title"])
            rows, cols = tile.shape
            # Each tile fills only its own part of the output; the rest keeps what is there.
            reproject(
                tile,
                values,
                src_transform=from_origin(
                    GRID_WEST + h * TILE_SIZE, GRID_NORTH - v * TILE_SIZE, TILE_SIZE / cols, TILE_SIZE / rows
                ),
                src_crs=CRS.from_proj4(SINUSOIDAL),
                src_nodata=fill,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                dst_nodata=fill,
                resampling=Resampling.nearest,
                init_dest_nodata=False,
            )

    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    return xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:4326")
