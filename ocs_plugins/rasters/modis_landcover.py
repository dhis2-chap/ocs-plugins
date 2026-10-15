"""MODIS land cover (MCD12Q1 v061, IGBP classes, 500 m, yearly) from NASA LP DAAC.

MCD12Q1 classifies each 500 m cell of the MODIS sinusoidal tile grid once a year, from 2001,
using Terra and Aqua. This plugin stores ``LC_Type1``, the 17-class IGBP scheme, as published,
reprojected to EPSG:4326 at ``resolution`` degrees (default 1/240, about 500 m). See ``_modis``
for how the tiles are found, read and mosaicked; nearest-neighbour resampling keeps the class
codes as they are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

from ocs_plugins.rasters import _modis

_SHORT_NAME = "MCD12Q1"
_VERSION = "061"
_FIELD = "LC_Type1"
# A small land area every year's product covers, to find the latest published year.
_PROBE_BBOX = [29.5, -2.0, 29.6, -1.9]
_FILL = 255
_FIRST_YEAR = 2001


def _latest_year() -> int | None:
    """The latest year with a published MCD12Q1 v061 map, from CMR."""
    latest = _modis.latest_granule_start(_SHORT_NAME, _VERSION, _PROBE_BBOX)
    return int(latest[:4]) if latest else None


def _read_classes(path: Path) -> np.ndarray:
    return _modis.read_fields(path, (_FIELD,))[_FIELD]


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
        year = period_id[:4]
        granules = _modis.granules_for_day(_SHORT_NAME, _VERSION, f"{year}-01-01", bbox)
        if not granules:
            raise RuntimeError(f"No MCD12Q1 land cover for {year} over {bbox}")
        classes = _modis.mosaic(granules, bbox, self.resolution, _read_classes, _FILL, "uint8")
        values = classes.where(classes != _FILL).astype("float32")
        values.attrs = {"units": "1", "flag_meanings": "IGBP class codes, see the template description"}
        ds = values.to_dataset(name="land_cover")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{year}-01-01", "ns")]}))
