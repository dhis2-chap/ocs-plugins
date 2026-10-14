"""NDVI from the Copernicus Land Monitoring Service (CLMS), global, 300 m, 10-daily (V3).

One composite per dekad (days 1-10, 11-20 and 21 to the end of the month), from PROBA-V
(2014-2018) and Sentinel-3 OLCI (2019 onwards), merged into one series. Each dekad is a single
global cloud-optimised GeoTIFF on a 1/336 degree EPSG:4326 grid, so only the window inside the
extent is read, over HTTP range requests, on the product's own grid.

Items are found through the Copernicus Data Space Ecosystem (CDSE) STAC API, which needs no
account. The files are read from the CDSE S3 store, which needs free S3 keys: from the
``CDSE_S3_ACCESS_KEY`` and ``CDSE_S3_SECRET_KEY`` environment variables, or else the ``cdse``
profile in ``~/.aws/credentials``.

Values 0-250 are NDVI (``0.004 * value - 0.08``); 252 (unknown), 253 (snow), 254 (water) and
255 (missing) become NaN.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, cast

import numpy as np
import requests
import xarray as xr
from open_climate_service.shared.time import dekad_period_ids
from open_climate_service.streaming import BaseDatasetPlugin

_STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
_COLLECTION = "clms_ndvi_global_300m_10daily_v3_cog"
_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
_S3_PROFILE = "cdse"
_KEY_ENV, _SECRET_ENV = "CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY"
_SCALE, _OFFSET = 0.004, -0.08
_MAX_VALID = 250
_NDVI_FILE = re.compile(r"Nodes\(([^)]*NDVI300-NDVI_[^)]*\.tiff)\)")


def _search(**params: Any) -> list[dict]:
    response = requests.get(_STAC_SEARCH, params={"collections": _COLLECTION, **params}, timeout=60)
    response.raise_for_status()
    return response.json()["features"]


def _ndvi_path(item: dict) -> str:
    """The GDAL path of an item's NDVI file in the CDSE S3 store.

    The S3 asset points at the file itself in older items, and at the product folder in newer
    ones; the file name is then taken from the HTTPS alternate.
    """
    asset = item["assets"]["ndvi"]
    if asset["href"].endswith(".tiff"):
        return "/vsis3/" + asset["href"].removeprefix("s3://")
    match = _NDVI_FILE.search(asset["alternate"]["https"]["href"])
    if match is None:
        raise RuntimeError(f"Cannot find the NDVI file name in CLMS item {item['id']}")
    return "/vsis3/" + asset["href"].removeprefix("s3://").rstrip("/") + "/" + match.group(1)


def _s3_options() -> dict[str, str]:
    options = {
        "AWS_S3_ENDPOINT": _S3_ENDPOINT,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
    }
    key, secret = os.getenv(_KEY_ENV), os.getenv(_SECRET_ENV)
    if key and secret:
        options |= {"AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    else:
        options["AWS_PROFILE"] = _S3_PROFILE
    return options


class CLMSNDVIDekadalPlugin(BaseDatasetPlugin):
    """Dekadal NDVI from the CLMS global 300 m V3 product."""

    async def periods(self, start: str, end: str) -> list[str]:
        latest = _search(sortby="-datetime", limit=1)
        if not latest:
            return []
        return dekad_period_ids(start, end, cutoff=latest[0]["properties"]["start_datetime"][:10])

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rasterio
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.errors import RasterioIOError
        from rasterio.windows import Window

        items = _search(datetime=f"{period_id}T00:00:00Z/{period_id}T23:59:59Z", limit=10)
        if not items:
            raise RuntimeError(f"No CLMS NDVI composite found for the dekad starting {period_id}")
        path = _ndvi_path(items[0])

        try:
            with rasterio.Env(**_s3_options()), rasterio.open(path) as src:
                # Widen the extent to whole pixels of the product's grid, then read just that window.
                inverse = ~src.transform
                col0, row0 = inverse * (bbox[0], bbox[3])
                col1, row1 = inverse * (bbox[2], bbox[1])
                col0, row0 = max(math.floor(col0), 0), max(math.floor(row0), 0)
                col1, row1 = min(math.ceil(col1), src.width), min(math.ceil(row1), src.height)
                window = Window(col0, row0, col1 - col0, row1 - row0)
                raw = src.read(1, window=window)
                transform = src.window_transform(window)
        except RasterioIOError as exc:
            if "403" in str(exc) or "AccessDenied" in str(exc) or "credentials" in str(exc).lower():
                raise PermissionError(
                    f"CDSE S3 refused {path}. Create free S3 keys at "
                    "https://eodata-s3keysmanager.dataspace.copernicus.eu and set "
                    f"{_KEY_ENV}/{_SECRET_ENV}, or add a [{_S3_PROFILE}] profile to ~/.aws/credentials."
                ) from exc
            raise

        ndvi = np.where(raw <= _MAX_VALID, raw.astype("float32") * np.float32(_SCALE) + np.float32(_OFFSET), np.nan)
        height, width = ndvi.shape
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        data = xr.DataArray(
            ndvi.astype("float32"), dims=("y", "x"), coords={"y": y, "x": x}, attrs={"units": "1", "long_name": "NDVI"}
        )
        ds = data.to_dataset(name="ndvi").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(period_id, "ns")]}))
