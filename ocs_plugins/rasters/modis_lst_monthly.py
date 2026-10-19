"""Monthly MODIS land surface temperature from MOD11A2 v061 on Microsoft Planetary Computer.

MOD11A2 (Terra) holds 8-day composites of day and night land surface temperature at 1 km. A
month is the mean of every composite overlapping it, each weighted by its days inside the month:
composites straddle month boundaries, and counting a two-day overlap as a full composite puts
much of a month's weight on days outside it.

This is the radiative skin temperature of the ground, not 2 m air temperature; on a daytime
overpass over dry ground it runs 10-20 °C warmer. It is clear-sky only, so a cloudy month has
fewer cells. Pixels whose QC error estimate exceeds 2 K are dropped, the usual threshold for
MOD11A2 aggregation. Tiles are reprojected to EPSG:4326 at ``resolution`` degrees (default
0.01, about 1 km). The Planetary Computer STAC API needs no account; the daily MOD11A1 in
``modis_lst`` comes from NASA LP DAAC and needs Earthdata Login.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, monthly_period_ids

from ocs_plugins.rasters.modis_vi import _latest_items

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "modis-11A2-061"
_SCALE_K = 0.02  # uint16 kelvin in 0.02 K steps; 0 is fill
_QC_ERROR_MAX = 1  # QC bits 6-7: 0 is an error of at most 1 K, 1 at most 2 K
_FIRST_MONTH = "2000-03"  # the first composite starts on 2000-02-18
# A month is offered once its last composite (ending up to 7 days after it) has been published.
_PUBLICATION_LAG = timedelta(days=30)


def _window(item: Any) -> tuple[date, date]:
    """The composite's first and last day."""
    start, end = (
        datetime.fromisoformat(item.properties[key].replace("Z", "+00:00")).date()
        for key in ("start_datetime", "end_datetime")
    )
    return start, end


def _days_in_month(item: Any, year: int, month: int) -> int:
    """How many of the composite's days fall in the month."""
    start, end = _window(item)
    first, last = date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    return max(0, (min(end, last) - max(start, first)).days + 1)


def _weighted_mean(lst: xr.DataArray, qc: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """The mean over `time` in °C of the good-quality pixels, each composite weighted by *weights*."""
    good = (lst > 0) & (((qc >> 6) & 0b11) <= _QC_ERROR_MAX)
    celsius = lst.astype("float32") * np.float32(_SCALE_K) - np.float32(273.15)
    total = (celsius.where(good, 0) * weights).sum("time")
    weight = (good * weights).sum("time")
    return cast(xr.DataArray, total / weight.where(weight > 0))


class MODISLandSurfaceTemperatureMonthlyPlugin(BaseDatasetPlugin):
    """Monthly mean day or night land surface temperature from MOD11A2.

    Template params:
        band: ``day`` or ``night``.
        resolution: output grid spacing in degrees (default 0.01).
    """

    max_concurrency = 1

    def __init__(self, band: str, resolution: float = 0.01, **_: Any) -> None:
        if band not in ("day", "night"):
            raise ValueError(f"{type(self).__name__}: band must be 'day' or 'night', got {band!r}")
        self.band = band
        self.resolution = float(resolution)

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = datetime.now(UTC).date() - _PUBLICATION_LAG
        last = cutoff if (cutoff + timedelta(days=1)).day == 1 else cutoff.replace(day=1) - timedelta(days=1)
        return monthly_period_ids(max(start[:7], _FIRST_MONTH), end, cutoff=last)

    def _search(self, year: int, month: int, bbox: list[float]) -> list[Any]:
        import planetary_computer
        import pystac_client

        window_start = date(year, month, 1) - timedelta(days=7)
        window_end = date(year, month, calendar.monthrange(year, month)[1])
        catalog = pystac_client.Client.open(_STAC_URL, modifier=planetary_computer.sign_inplace)
        found = catalog.search(collections=[_COLLECTION], bbox=bbox, datetime=f"{window_start}/{window_end}").items()
        # The collection also holds Aqua's MYD11A2, from a 13:30 rather than a 10:30 overpass.
        terra = [item for item in found if item.id.startswith("MOD11A2.") and _days_in_month(item, year, month) > 0]
        return _latest_items(terra)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import odc.stac
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor

        year, month = int(period_id[:4]), int(period_id[5:7])
        items = self._search(year, month, bbox)
        if not items:
            raise RuntimeError(f"No MOD11A2 composites found for {period_id} over {bbox}")
        lst_band, qc_band = f"LST_{self.band.capitalize()}_1km", f"QC_{self.band.capitalize()}"
        loaded = odc.stac.load(
            items,
            bands=[lst_band, qc_band],
            crs="EPSG:4326",
            resolution=self.resolution,
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            groupby="solar_day",  # the tiles of one composite share its start day
            chunks={"x": 2048, "y": 2048},
            resampling="nearest",
            dtype="uint16",
            nodata=0,
        )
        days = {_window(item)[0]: _days_in_month(item, year, month) for item in items}
        weights = xr.DataArray(
            [days[np.datetime64(t, "D").astype(date)] for t in loaded.time.values],
            dims="time",
            coords={"time": loaded.time},
        )
        monthly = _weighted_mean(loaded[lst_band], loaded[qc_band], weights).compute()
        monthly = monthly.rename({"latitude": "y", "longitude": "x"}).drop_vars("spatial_ref", errors="ignore")
        variable = f"lst_{self.band}"
        monthly = monthly.assign_attrs(units="degC", long_name=f"MODIS {self.band}time land surface temperature")
        ds = monthly.to_dataset(name=variable).rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
