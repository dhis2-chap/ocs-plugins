"""Gross and net primary production from the Copernicus Land Monitoring Service, 300 m, 10-daily.

One value per dekad (days 1-10, 11-20 and 21 to the end of the month) from November 2013, from
PROBA-V and then Sentinel-3 OLCI, in grams of carbon per square metre per day. CLMS derives NPP
from GPP with a constant carbon use efficiency of 0.5, so NPP is half of GPP in every pixel and
never negative. Read from the Copernicus Data Space Ecosystem; see ``_clms`` for access and
credentials.

CLMS publishes each dekad in near-real-time versions RT0, RT1 and RT2 within days, then a final
consolidated RT6 about two months later. OCS stores are appended to and not revised, so only
dekads whose RT6 is out are offered; a dekad is never stored from a preliminary version.

Negative raw values are flags (-1 missing, -2 sea) and become NaN; the rest are scaled with the
file's own scale factor.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.shared.time import dekad_period_ids
from open_climate_service.streaming import BaseDatasetPlugin

from ocs_plugins.rasters import _clms

_COLLECTIONS = {
    "gpp": "clms_gpp_global_300m_10daily_v2_cog",
    "npp": "clms_npp_global_300m_10daily_v2_cog",
}
_NAMES = {"gpp": "Gross primary production", "npp": "Net primary production"}
_FINAL = 6
_RT = re.compile(r"-RT(\d+)_")
# Enough recent items to reach back past the near-real-time versions to the latest final one.
_RECENT_ITEMS = 100


def _rt(item: dict) -> int:
    match = _RT.search(item["id"])
    return int(match.group(1)) if match else _FINAL


class CLMSProductivityDekadalPlugin(BaseDatasetPlugin):
    """Dekadal GPP or NPP from the CLMS global 300 m V2 products.

    Template params:
        product: ``gpp`` or ``npp``.
    """

    def __init__(self, product: str, **_: Any) -> None:
        if product not in _COLLECTIONS:
            raise ValueError(
                f"{type(self).__name__}: unsupported product {product!r}; expected one of {list(_COLLECTIONS)}"
            )
        self.product = product
        self.collection = _COLLECTIONS[product]

    async def periods(self, start: str, end: str) -> list[str]:
        # The catalogue lookup is a blocking HTTP call; keep it off the event loop.
        recent = await asyncio.to_thread(_clms.search, self.collection, sortby="-datetime", limit=_RECENT_ITEMS)
        final = [item["properties"]["start_datetime"][:10] for item in recent if _rt(item) == _FINAL]
        if not final:
            return []
        return dekad_period_ids(start, end, cutoff=max(final))

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        items = _clms.search(self.collection, datetime=f"{period_id}T00:00:00Z/{period_id}T23:59:59Z", limit=20)
        final = [item for item in items if _rt(item) == _FINAL]
        if not final:
            raise RuntimeError(f"No final (RT6) CLMS {self.product.upper()} for the dekad starting {period_id}")
        values = _clms.read_window(_clms.asset_path(final[0], self.product), bbox, lambda raw: raw >= 0)
        values = values.assign_attrs(units="g m-2 day-1", long_name=_NAMES[self.product])
        return cast(
            xr.Dataset, values.to_dataset(name=self.product).expand_dims({"t": [np.datetime64(period_id, "ns")]})
        )
