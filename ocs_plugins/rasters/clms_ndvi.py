"""NDVI from the Copernicus Land Monitoring Service (CLMS), global, 300 m, 10-daily (V3).

One composite per dekad (days 1-10, 11-20 and 21 to the end of the month), from PROBA-V
(2014-2018) and Sentinel-3 OLCI (2019 onwards), merged into one series. Read from the Copernicus
Data Space Ecosystem; see ``_clms`` for access and credentials.

Values 0-250 are NDVI (``0.004 * value - 0.08``, from the file's scale and offset); 252
(unknown), 253 (snow), 254 (water) and 255 (missing) become NaN.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.shared.time import dekad_period_ids
from open_climate_service.streaming import BaseDatasetPlugin

from ocs_plugins.rasters import _clms

_COLLECTION = "clms_ndvi_global_300m_10daily_v3_cog"
_MAX_VALID = 250


class CLMSNDVIDekadalPlugin(BaseDatasetPlugin):
    """Dekadal NDVI from the CLMS global 300 m V3 product."""

    async def periods(self, start: str, end: str) -> list[str]:
        latest = _clms.search(_COLLECTION, sortby="-datetime", limit=1)
        if not latest:
            return []
        return dekad_period_ids(start, end, cutoff=latest[0]["properties"]["start_datetime"][:10])

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        items = _clms.search(_COLLECTION, datetime=f"{period_id}T00:00:00Z/{period_id}T23:59:59Z", limit=10)
        if not items:
            raise RuntimeError(f"No CLMS NDVI composite found for the dekad starting {period_id}")
        ndvi = _clms.read_window(_clms.asset_path(items[0], "ndvi"), bbox, lambda raw: raw <= _MAX_VALID)
        ndvi = ndvi.assign_attrs(units="1", long_name="NDVI")
        return cast(xr.Dataset, ndvi.to_dataset(name="ndvi").expand_dims({"t": [np.datetime64(period_id, "ns")]}))
