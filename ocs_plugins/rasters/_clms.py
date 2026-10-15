"""Copernicus Land Monitoring Service (CLMS) products on the Copernicus Data Space Ecosystem.

Shared by the CLMS plugins. Items are found through the CDSE STAC API, which needs no account.
Each item is one global cloud-optimised GeoTIFF on a 1/336 degree EPSG:4326 grid in the CDSE S3
store; only the window inside the extent is read, over HTTP range requests, on that grid.

Reading needs free CDSE S3 keys: from the ``CDSE_S3_ACCESS_KEY`` and ``CDSE_S3_SECRET_KEY``
environment variables, or else the ``cdse`` profile in ``~/.aws/credentials``. They reach GDAL
through a rasterio ``AWSSession`` scoped to each read, so parallel fetches and other plugins do
not share them, and nothing is written to the process environment.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
import xarray as xr

STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
_S3_PROFILE = "cdse"
_KEY_ENV, _SECRET_ENV = "CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY"
_NODE = re.compile(r"Nodes\(([^)]*\.tiff?)\)")


def search(collection: str, **params: Any) -> list[dict]:
    """STAC items of ``collection`` matching ``params`` (no account needed)."""
    response = requests.get(STAC_SEARCH, params={"collections": collection, **params}, timeout=60)
    response.raise_for_status()
    return response.json()["features"]


def asset_path(item: dict, variable: str) -> str:
    """The GDAL path of an item's ``variable`` file in the CDSE S3 store.

    The asset is keyed by the variable (``ndvi``) or prefixed with the product (``npp300_npp``).
    Its S3 href is the file itself in some items and the product folder in others; the file name
    is then taken from the HTTPS alternate.
    """
    keys = [key for key in item["assets"] if key == variable or key.endswith(f"_{variable}")]
    if not keys:
        raise RuntimeError(f"CLMS item {item['id']} has no {variable!r} asset; it has {sorted(item['assets'])}")
    asset = item["assets"][keys[0]]
    href = asset["href"]
    if href.endswith((".tif", ".tiff")):
        return "/vsis3/" + href.removeprefix("s3://")
    match = _NODE.search(asset.get("alternate", {}).get("https", {}).get("href", ""))
    if match is None:
        raise RuntimeError(f"Cannot find the {variable} file name in CLMS item {item['id']}")
    return "/vsis3/" + href.removeprefix("s3://").rstrip("/") + "/" + match.group(1)


_NO_KEYS = (
    "Create free CDSE S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu and set "
    f"{_KEY_ENV}/{_SECRET_ENV}, or add a [{_S3_PROFILE}] profile to ~/.aws/credentials."
)


def _s3_session() -> Any:
    """A rasterio session for the CDSE S3 store; rasterio takes S3 keys only this way."""
    from botocore.exceptions import ProfileNotFound
    from rasterio.session import AWSSession

    key, secret = os.getenv(_KEY_ENV), os.getenv(_SECRET_ENV)
    if key and secret:
        return AWSSession(aws_access_key_id=key, aws_secret_access_key=secret, endpoint_url=_S3_ENDPOINT)
    try:
        return AWSSession(profile_name=_S3_PROFILE, endpoint_url=_S3_ENDPOINT)
    except ProfileNotFound as exc:
        raise PermissionError(f"No CDSE S3 keys. {_NO_KEYS}") from exc


def read_window(path: str, bbox: list[float], valid: Callable[[np.ndarray], np.ndarray]) -> xr.DataArray:
    """Read ``bbox`` from a CLMS file on its own grid, scaled to physical values.

    The extent is widened to whole pixels. Raw values for which ``valid`` is false (the product's
    flags) become NaN; the rest are scaled with the file's own scale and offset.
    """
    import rasterio
    import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
    from rasterio.errors import RasterioIOError
    from rasterio.windows import Window

    try:
        with (
            rasterio.Env(
                _s3_session() if path.startswith("/vsis3/") else None,
                AWS_VIRTUAL_HOSTING="FALSE",
                AWS_HTTPS="YES",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
            ),
            rasterio.open(path) as src,
        ):
            inverse = ~src.transform
            col0, row0 = inverse * (bbox[0], bbox[3])
            col1, row1 = inverse * (bbox[2], bbox[1])
            col0, row0 = max(math.floor(col0), 0), max(math.floor(row0), 0)
            col1, row1 = min(math.ceil(col1), src.width), min(math.ceil(row1), src.height)
            window = Window(col0, row0, col1 - col0, row1 - row0)
            raw = src.read(1, window=window)
            transform = src.window_transform(window)
            scale, offset = src.scales[0], src.offsets[0]
    except RasterioIOError as exc:
        message = str(exc)
        if "403" in message or "AccessDenied" in message or "InvalidAccessKeyId" in message:
            raise PermissionError(f"CDSE S3 refused {path}. {_NO_KEYS}") from exc
        raise

    values = np.where(valid(raw), raw.astype("float32") * np.float32(scale) + np.float32(offset), np.nan)
    height, width = values.shape
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    return xr.DataArray(values.astype("float32"), dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:4326")
