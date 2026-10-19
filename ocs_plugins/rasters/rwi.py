"""Meta Relative Wealth Index for one country, on its native zoom-14 Web Mercator grid.

HDX publishes each country as a CSV of cell centres with the unpopulated cells omitted. The
points are a regular grid in EPSG:3857 only (Mercator y is not linear in latitude), so the grid
is rebuilt there and stored in that CRS; empty cells are NaN, never 0, which would read as
"poorest". Values rank areas within one country only.

The file is found by the ISO3 stem of its name. 16 of the 109 files are named by country slug,
and India and Pakistan share one; set `resource_filename` for those.
"""

from __future__ import annotations

import io
import math
from typing import Any

import httpx
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

_HDX_PACKAGE = "https://data.humdata.org/api/3/action/package_show?id=relative-wealth-index"
_SUFFIX = "_relative_wealth_index.csv"
_YEAR = "2021"  # the one release, April 2021
_RADIUS = 6378137.0
_CELL_M = 2 * math.pi * _RADIUS / 2**14  # a zoom-14 Bing tile, about 2446 m


def _resource_url(filename: str) -> str:
    resources = httpx.get(_HDX_PACKAGE, timeout=120, follow_redirects=True).raise_for_status().json()
    urls = {str(r["url"]).rsplit("/", 1)[-1]: str(r["url"]) for r in resources["result"]["resources"]}
    if filename not in urls:
        raise ValueError(
            f"No Relative Wealth Index file {filename!r}: the release covers low- and middle-income "
            "countries only, and names some by country slug; set `resource_filename` for those"
        )
    return urls[filename]


class RelativeWealthIndexPlugin(BaseDatasetPlugin):
    """The index for the instance's `extent.country_code`, or for `resource_filename`."""

    crs = 3857

    def __init__(self, resource_filename: str | None = None, **_: Any) -> None:
        self.resource_filename = resource_filename

    async def periods(self, start: str, end: str) -> list[str]:
        return [_YEAR] if int(start[:4]) <= int(_YEAR) <= int(end[:4]) else []

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        country = params.get("country_code")
        if not self.resource_filename and not country:
            raise ValueError("RWI needs the instance's `extent.country_code`, or `resource_filename`")
        filename = self.resource_filename or f"{str(country).lower()}{_SUFFIX}"
        response = httpx.get(_resource_url(filename), timeout=300, follow_redirects=True).raise_for_status()
        points = pd.read_csv(io.BytesIO(response.content))

        # Every cell that overlaps the extent, not only those centred in it: half a cell is at most
        # 180 / 2**14 degrees in either direction.
        west, south, east, north = bbox
        pad = 180 / 2**14
        points = points[
            points.longitude.between(west - pad, east + pad) & points.latitude.between(south - pad, north + pad)
        ]
        if points.empty:
            raise ValueError(f"No RWI points from {filename} fall inside the extent {bbox}")

        # Web Mercator is spherical by definition, so the closed form matches the tile grid exactly.
        x = np.radians(points.longitude.to_numpy()) * _RADIUS
        y = np.log(np.tan(np.pi / 4 + np.radians(points.latitude.to_numpy()) / 2)) * _RADIUS
        ix = np.round((x - x.min()) / _CELL_M).astype(int)
        iy = np.round((y.max() - y) / _CELL_M).astype(int)  # rows from the north, as OCS expects

        grid = np.full((iy.max() + 1, ix.max() + 1), np.nan, dtype="float32")
        grid[iy, ix] = points.rwi.to_numpy(dtype="float32")
        rwi = xr.DataArray(
            grid[np.newaxis],
            coords={
                "t": [np.datetime64(f"{_YEAR}-01-01", "ns")],
                "y": y.max() - np.arange(grid.shape[0]) * _CELL_M,
                "x": x.min() + np.arange(grid.shape[1]) * _CELL_M,
            },
            dims=("t", "y", "x"),
            attrs={
                "long_name": "Meta Relative Wealth Index",
                "units": "1",
                "references": "Chi, Fang, Chatterjee & Blumenstock, PNAS 2022, doi:10.1073/pnas.2113658119",
            },
        )
        return rwi.to_dataset(name="rwi").rio.write_crs(3857)
