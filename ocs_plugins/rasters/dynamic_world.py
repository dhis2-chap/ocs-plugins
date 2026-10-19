"""Dynamic World land cover, monthly, from Google Earth Engine.

Dynamic World (Google and the World Resources Institute) classifies every Sentinel-2 scene with at
most 35 % cloud into nine classes at 10 m, within days of acquisition, from June 2015 on. A scene
covers only part of a country and its clouds are masked, so one month's map is the most frequent
class of each cell across that month's scenes. A cell that no scene saw clearly is no data, stored
as missing (NaN) since class 0 is water. A month is offered once its last scenes have had a week to
arrive.

The output grid is in degrees, ``pixel_degrees`` wide, 0.001 (about 110 m) by default: at 10 m a
country is hundreds of millions of cells per month. The grid is requested at that size, so Earth
Engine reads each scene from its lower-resolution levels, which for the class band keep the most
frequent 10 m class (its pyramiding policy is MODE). That is the scene's most frequent class within
each cell, without computing the month at 10 m first, which runs out of Earth Engine's memory for a
country. The grid is read with Xee, which splits it into requests within Earth Engine's size limit
and fetches them in parallel from the high-volume endpoint.

Earth Engine needs an account and a Google Cloud project registered for it. Credentials are the
ones the Earth Engine client finds itself (``earthengine authenticate``, or a service account
through ``GOOGLE_APPLICATION_CREDENTIALS``), and ``EARTHENGINE_PROJECT`` names the project.
"""

from __future__ import annotations

import calendar
import math
import os
import threading
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401  # activates the .rio accessor used below
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
_FIRST_MONTH = "2015-07"  # the record starts on 27 June 2015, so July is the first full month
_NO_DATA = 255  # Earth Engine's code for a cell with no scene, read back as missing
# Scenes reach Earth Engine days after acquisition, and a stored month is not read again (append),
# so a month is offered only once its last scenes have had this long to arrive.
_SETTLE = timedelta(days=7)
# The endpoint Earth Engine provides for many parallel automated requests, which Xee makes.
_HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
_SCOPES = ["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"]

_init_lock = threading.Lock()
_initialised = False


def _ee() -> Any:
    """The Earth Engine client, initialised once with ``EARTHENGINE_PROJECT``."""
    global _initialised
    import ee

    with _init_lock:
        if not _initialised:
            project = os.environ.get("EARTHENGINE_PROJECT")
            if not project:
                raise RuntimeError(
                    "Dynamic World needs EARTHENGINE_PROJECT set to a Google Cloud project registered for Earth Engine"
                )
            # The client does not read a service account key by itself: it only finds the
            # credentials `earthengine authenticate` stored ("persistent"), so a key file is loaded here.
            credentials: Any = "persistent"
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                import google.auth

                credentials, _ = google.auth.default(scopes=_SCOPES)
            ee.Initialize(credentials, project=project, opt_url=_HIGH_VOLUME)
            _initialised = True
    return ee


def _months(start: str, end: str, today: date | None = None) -> list[str]:
    """Months from ``start`` to ``end`` that are complete, settled and within the record, as ``YYYY-MM``."""
    settled = (today or date.today()) - _SETTLE
    last_complete = (settled.year, settled.month - 1) if settled.month > 1 else (settled.year - 1, 12)
    first = max((int(start[:4]), int(start[5:7])), (int(_FIRST_MONTH[:4]), int(_FIRST_MONTH[5:7])))
    last = min((int(end[:4]), int(end[5:7])), last_complete)
    months = []
    year, month = first
    while (year, month) <= last:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    return months


def _grid(bbox: list[float], pixel: float) -> tuple[float, float, int, int]:
    """West and north edges, and width and height in cells, of ``bbox`` widened to whole cells."""
    west = math.floor(round(bbox[0] / pixel, 6)) * pixel
    north = math.ceil(round(bbox[3] / pixel, 6)) * pixel
    width = math.ceil(round((bbox[2] - west) / pixel, 6))
    height = math.ceil(round((north - bbox[1]) / pixel, 6))
    return west, north, width, height


class DynamicWorldPlugin(BaseDatasetPlugin):
    """Most frequent Dynamic World class of each cell in a month (0 water ... 8 snow and ice; NaN no data)."""

    # One month at a time; Xee already reads each month in parallel requests.
    max_concurrency = 1

    def __init__(self, pixel_degrees: float = 0.001) -> None:
        self._pixel = float(pixel_degrees)

    async def periods(self, start: str, end: str) -> list[str]:
        return _months(start, end)

    def _composite(self, period_id: str, bbox: list[float]) -> Any:
        """The month's most frequent class of each cell, as a one-image collection; None if no scene covers ``bbox``."""
        ee = _ee()
        year, month = int(period_id[:4]), int(period_id[5:7])
        first = f"{period_id}-01"
        after = date(year, month, calendar.monthrange(year, month)[1]).toordinal() + 1
        region = ee.Geometry.Rectangle(bbox)
        scenes = (
            ee.ImageCollection(_COLLECTION)
            .filterDate(first, date.fromordinal(after).isoformat())
            .filterBounds(region)
            .select("label")
        )
        if scenes.size().getInfo() == 0:
            return None
        month = scenes.mode().unmask(_NO_DATA).toUint8()
        return ee.ImageCollection([month.set("system:time_start", ee.Date(first).millis())])

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        west, north, width, height = _grid(bbox, self._pixel)
        classes = np.full((height, width), np.nan, dtype="float32")
        month = self._composite(period_id, bbox)
        if month is not None:
            read = xr.open_dataset(
                month,
                engine="ee",
                crs="EPSG:4326",
                crs_transform=(self._pixel, 0, west, 0, -self._pixel, north),
                shape_2d=(width, height),
            )
            # Missing, not a code: a cell that is no class counts towards none in the statistics.
            classes = read["label"].isel(time=0).transpose("y", "x").values.astype("float32")
            classes[classes == _NO_DATA] = np.nan

        x = west + (np.arange(width) + 0.5) * self._pixel
        y = north - (np.arange(height) + 0.5) * self._pixel
        landcover = xr.DataArray(classes, dims=("y", "x"), coords={"y": y, "x": x}, attrs={"units": "1"})
        ds = landcover.to_dataset(name="landcover").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
