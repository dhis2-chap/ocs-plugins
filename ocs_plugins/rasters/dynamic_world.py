"""Dynamic World land cover, monthly, from Google Earth Engine.

Dynamic World (Google and the World Resources Institute) classifies every Sentinel-2 scene with at
most 35 % cloud into nine classes at 10 m, within days of acquisition, from June 2015 on. A scene
covers only part of a country and its clouds are masked, so one month's map is the most frequent
class of each cell across that month's scenes. A cell that no scene saw clearly is no data (255),
since class 0 is water.

The output grid is in degrees, ``pixel_degrees`` wide, 0.001 (about 110 m) by default: at 10 m a
country is hundreds of millions of cells per month. When the grid is coarser than 10 m, Earth
Engine takes the most frequent 10 m class within each cell rather than sampling one pixel. The
grid is fetched in tiles with ``computePixels``, each within its request size limit.

Earth Engine needs an account and a Google Cloud project registered for it. Credentials are the
ones the Earth Engine client finds itself (``earthengine authenticate``, or a service account
through ``GOOGLE_APPLICATION_CREDENTIALS``), and ``EARTHENGINE_PROJECT`` names the project.
"""

from __future__ import annotations

import calendar
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401  # activates the .rio accessor used below
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
_FIRST_MONTH = "2015-07"  # the record starts on 27 June 2015, so July is the first full month
_NATIVE_METRES = 10
_NO_DATA = 255
_TILE = 4096  # cells per tile side: 16 MB of uint8, well within computePixels' 48 MB
_WORKERS = 4

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
            ee.Initialize(project=project)
            _initialised = True
    return ee


def _months(start: str, end: str) -> list[str]:
    """Months from ``start`` to ``end`` that are complete and within the record, as ``YYYY-MM``."""
    today = date.today()
    last_complete = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
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


def _tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """``(column, row, width, height)`` of the tiles covering a ``width`` x ``height`` grid."""
    return [
        (col, row, min(_TILE, width - col), min(_TILE, height - row))
        for row in range(0, height, _TILE)
        for col in range(0, width, _TILE)
    ]


class DynamicWorldPlugin(BaseDatasetPlugin):
    """Most frequent Dynamic World class of each cell in a month (0 water ... 8 snow and ice; 255 no data)."""

    # One month at a time; its tiles are fetched a few at a time, well within Earth Engine's request quota.
    max_concurrency = 1

    def __init__(self, pixel_degrees: float = 0.001) -> None:
        self._pixel = float(pixel_degrees)

    async def periods(self, start: str, end: str) -> list[str]:
        return _months(start, end)

    def _composite(self, period_id: str, bbox: list[float]) -> Any:
        """The month's most frequent class, aggregated to the output grid; None if no scene covers ``bbox``."""
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
        # A composite has no projection of its own; give it the native 10 m, so a coarser grid
        # takes the most frequent class of the 10 m cells within each cell.
        composite = scenes.mode().setDefaultProjection(ee.Projection("EPSG:4326").atScale(_NATIVE_METRES))
        cell_metres = self._pixel * 111_320
        if cell_metres > _NATIVE_METRES * 1.5:
            per_cell = math.ceil(cell_metres / _NATIVE_METRES) ** 2
            composite = composite.reduceResolution(ee.Reducer.mode(), maxPixels=min(65535, per_cell + 64))
        return composite.unmask(_NO_DATA).toUint8()

    def _tile(self, composite: Any, west: float, north: float, tile: tuple[int, int, int, int]) -> np.ndarray:
        col, row, width, height = tile
        ee = _ee()
        pixels = ee.data.computePixels(
            {
                "expression": composite,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": width, "height": height},
                    "affineTransform": {
                        "scaleX": self._pixel,
                        "shearX": 0,
                        "translateX": west + col * self._pixel,
                        "shearY": 0,
                        "scaleY": -self._pixel,
                        "translateY": north - row * self._pixel,
                    },
                    "crsCode": "EPSG:4326",
                },
            }
        )
        return np.asarray(pixels[pixels.dtype.names[0]] if pixels.dtype.names else pixels, dtype="uint8")

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        west, north, width, height = _grid(bbox, self._pixel)
        classes = np.full((height, width), _NO_DATA, dtype="uint8")
        composite = self._composite(period_id, bbox)
        if composite is not None:
            tiles = _tiles(width, height)
            with ThreadPoolExecutor(_WORKERS) as pool:
                parts = pool.map(lambda tile: self._tile(composite, west, north, tile), tiles)
                for (col, row, w, h), part in zip(tiles, parts, strict=True):
                    classes[row : row + h, col : col + w] = part

        x = west + (np.arange(width) + 0.5) * self._pixel
        y = north - (np.arange(height) + 0.5) * self._pixel
        landcover = xr.DataArray(classes, dims=("y", "x"), coords={"y": y, "x": x}, attrs={"units": "1"})
        landcover.encoding = {"_FillValue": np.uint8(_NO_DATA)}
        ds = landcover.to_dataset(name="landcover").rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
