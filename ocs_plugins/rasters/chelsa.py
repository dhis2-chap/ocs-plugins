"""CHELSA v2.1 monthly mean near-surface air temperature at 30 arc-seconds (about 1 km).

CHELSA downscales ERA5 with a terrain-aware model, which matters in mountainous areas where
ERA5-Land's 9 km cells average over very different elevations. The monthly series runs from
January 1980 to December 2021 and is no longer extended.

Each month is one global cloud-optimised GeoTIFF on the CHELSA object store, in tenths of a
kelvin (65535 is no data); only the window inside the extent is read, on the product's own grid,
and converted to degrees Celsius. No account is needed.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, monthly_period_ids

_URL = (
    "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/monthly/tas/{year}/CHELSA_tas_{month:02d}_{year}_V.2.1.tif"
)
_FIRST, _LAST = "1980-01", "2021-12"
_NO_DATA = 65535
_KELVIN = 273.15


def _path(year: int, month: int) -> str:
    return "/vsicurl/" + _URL.format(year=year, month=month)


class CHELSATemperaturePlugin(BaseDatasetPlugin):
    """Monthly mean near-surface air temperature (degrees Celsius) from CHELSA v2.1."""

    max_concurrency = 2

    async def periods(self, start: str, end: str) -> list[str]:
        return monthly_period_ids(max(start[:7], _FIRST), end[:7], cutoff=_LAST)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rasterio
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.windows import Window

        year, month = int(period_id[:4]), int(period_id[5:7])
        with (
            rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"),
            rasterio.open(_path(year, month)) as src,
        ):
            # Widen the extent to whole pixels of the product's grid, then read just that window.
            inverse = ~src.transform
            col0, row0 = inverse * (bbox[0], bbox[3])
            col1, row1 = inverse * (bbox[2], bbox[1])
            col0, row0 = max(math.floor(col0), 0), max(math.floor(row0), 0)
            col1, row1 = min(math.ceil(col1), src.width), min(math.ceil(row1), src.height)
            window = Window(col0, row0, col1 - col0, row1 - row0)
            raw = src.read(1, window=window)
            transform = src.window_transform(window)
            scale, offset = src.scales[0], src.offsets[0]

        kelvin = raw.astype("float32") * np.float32(scale) + np.float32(offset)
        celsius = np.where(raw == _NO_DATA, np.nan, kelvin - np.float32(_KELVIN)).astype("float32")
        height, width = celsius.shape
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        data = xr.DataArray(celsius, dims=("y", "x"), coords={"y": y, "x": x}, attrs={"units": "degC"})
        ds = data.to_dataset(name="temperature").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
