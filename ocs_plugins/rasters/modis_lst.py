"""Daily daytime land surface temperature from MODIS Terra (MOD11A1 v061, 1 km) via NASA LP DAAC.

Land surface temperature is how hot the ground is, not the air: hotter than air temperature by
day, especially on bare and built-up land. It is observed only under clear sky, so cloudy cells
are empty.

Each day's tiles give ``LST_Day_1km`` (kelvin as 0.02 x stored value, 0 is fill) and the
``QC_Day`` quality bitfield. A cell is kept when the mandatory quality flag says good quality
(bits 0-1 are 0) and the average LST error is at most 2 K (bits 6-7 are 0 or 1); the value is
converted to degrees Celsius. See ``_modis`` for how tiles are found, read and mosaicked.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, daily_period_ids

from ocs_plugins.rasters import _modis
from ocs_plugins.rasters.earthdata import cmr_granules

_SHORT_NAME = "MOD11A1"
_VERSION = "061"
_LST, _QC = "LST_Day_1km", "QC_Day"
_SCALE = 0.02
_KELVIN = 273.15
_FIRST_DAY = "2000-02-24"
# A small land area, to find the latest published day.
_PROBE_BBOX = [29.5, -2.0, 29.6, -1.9]


def _latest_day() -> date | None:
    """The latest day with a published MOD11A1 v061 granule, from CMR."""
    today = datetime.now(UTC).date()
    granules = cmr_granules(
        _SHORT_NAME,
        (f"{today - timedelta(days=30)}T00:00:00Z", f"{today}T23:59:59Z"),
        bbox=_PROBE_BBOX,
        version=_VERSION,
    )
    return max((date.fromisoformat(granule["time_start"][:10]) for granule in granules), default=None)


def good_quality(qc: np.ndarray) -> np.ndarray:
    """Cells whose mandatory QA is good (bits 0-1 = 0) and LST error is at most 2 K (bits 6-7 <= 1)."""
    qc = qc.astype("int64")
    return ((qc & 0b11) == 0) & (((qc >> 6) & 0b11) <= 1)


def _read_celsius(path: Path) -> np.ndarray:
    fields = _modis.read_fields(path, (_LST, _QC))
    raw = fields[_LST]
    keep = good_quality(fields[_QC]) & (raw > 0)
    return np.where(keep, raw.astype("float32") * np.float32(_SCALE) - np.float32(_KELVIN), np.nan).astype("float32")


class MODISLSTDayPlugin(BaseDatasetPlugin):
    """Daily good-quality daytime land surface temperature (degrees Celsius) from MOD11A1.

    Template params:
        resolution: output grid spacing in degrees (default 1/120, about 1 km).
    """

    max_concurrency = 2
    commit_batch_size = 31

    def __init__(self, resolution: float = 1 / 120, **_: Any) -> None:
        self.resolution = float(resolution)

    async def periods(self, start: str, end: str) -> list[str]:
        import asyncio

        latest = await asyncio.to_thread(_latest_day)
        if latest is None:
            return []
        return daily_period_ids(max(start[:10], _FIRST_DAY), end, cutoff=latest)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = period_id[:10]
        granules = _modis.granules_for_day(_SHORT_NAME, _VERSION, day, bbox)
        if not granules:
            raise RuntimeError(f"No MOD11A1 land surface temperature for {day} over {bbox}")
        celsius = _modis.mosaic(granules, bbox, self.resolution, _read_celsius, np.nan, "float32")
        celsius.attrs = {"units": "degC", "long_name": "daytime land surface temperature"}
        ds = celsius.to_dataset(name="lst_day")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(day, "ns")]}))
