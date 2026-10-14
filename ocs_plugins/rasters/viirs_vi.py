"""VIIRS vegetation indices (NDVI, EVI) from VNP13A1/VJ113A1/VJ213A1 v002 on NASA Earthdata.

The 16-day composites at 500 m come from three satellites carrying VIIRS: Suomi NPP
(VNP13A1, from 2012), NOAA-20 (VJ113A1, from 2018) and NOAA-21 (VJ213A1, from 2023). VIIRS
continues the MODIS vegetation index record, which stops when NASA ends MODIS science
collection in 2027.

OCS has no 16-day period type, so this plugin publishes monthly values: the mean of every
good-quality composite whose compositing window has its midpoint in the month. The satellites'
composites are staggered, so a month gets several composites rather than two, which matters
when one is missing.

Pixels are kept when the pixel reliability rank is 0 to 3 (excellent, good, acceptable or
marginal); pass, questionable, poor, cloud shadow, snow, cloud and fill are dropped before
averaging. The VIIRS ranks are finer than MODIS's, so these are its good and marginal classes.

The tiles are on the MODIS sinusoidal grid and are reprojected to EPSG:4326 at ``resolution``
degrees (default 0.005, about 500 m), with nearest-neighbour resampling. Granules are found
through CMR and downloaded from LP DAAC with the Earthdata Login credentials in ``~/.netrc``.
"""

from __future__ import annotations

import calendar
import math
import re
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin, monthly_period_ids

from ocs_plugins._netcdf import NETCDF_LOCK
from ocs_plugins.rasters.earthdata import cmr_granules, data_links, download, earthdata_session

_PRODUCTS = ("VNP13A1", "VJ113A1", "VJ213A1")
_FIELDS = {"ndvi": "500 m 16 days NDVI", "evi": "500 m 16 days EVI"}
_RELIABILITY_FIELD = "500 m 16 days pixel reliability"
# Ranks 0-3: excellent, good, acceptable, marginal (VNP13 user guide, table 5).
_KEPT_RANKS = (0, 3)
_SCALE = 1e-4
# Valid VI range is -10000..10000 (scaled); the fills, -15000 over water and -13000 over land, lie outside it.
_VALID_RANGE = (-10000, 10000)
# A month is offered once the last composite whose midpoint falls in it has had time to be
# published: compositing ends up to ~8 days after the month, plus production latency.
_PUBLICATION_LAG = timedelta(days=30)

# The MODIS sinusoidal tile grid, which the VIIRS land products share.
_SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
_TILE_SIZE = 1111950.5196666666
_GRID_WEST = -20015109.354
_GRID_NORTH = 10007554.677
_TILE = re.compile(r"\.h(\d{2})v(\d{2})\.")


def _midpoint(granule: dict) -> date:
    start = datetime.fromisoformat(granule["time_start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(granule["time_end"].replace("Z", "+00:00"))
    return (start + (end - start) / 2).date()


def _latest_granules(granules: list[dict]) -> list[dict]:
    """Keep one granule per (product, composite date, tile): the most recently produced.

    Titles look like ``VNP13A1.A2026249.h21v09.002.2026265092155``; the last part is the
    production time, so a reprocessed granule wins over the original.
    """
    newest: dict[tuple[str, str, str], dict] = {}
    for granule in granules:
        product, composite, tile, _, produced = granule["title"].split(".")[:5]
        key = (product, composite, tile)
        if key not in newest or produced > newest[key]["title"].split(".")[4]:
            newest[key] = granule
    return list(newest.values())


def _find_variable(group: Any, name: str) -> Any:
    """The variable called ``name`` anywhere below ``group``."""
    if name in group.variables:
        return group.variables[name]
    for child in group.groups.values():
        found = _find_variable(child, name)
        if found is not None:
            return found
    return None


def _read_tile(path: Path, field: str) -> tuple[np.ndarray, np.ndarray]:
    """The raw index and pixel reliability arrays of one granule."""
    import netCDF4

    with NETCDF_LOCK, netCDF4.Dataset(path) as nc:
        nc.set_auto_maskandscale(False)
        arrays = []
        for name in (field, _RELIABILITY_FIELD):
            variable = _find_variable(nc, name)
            if variable is None:
                raise RuntimeError(f"{path.name} has no {name!r} field; the VIIRS file layout may have changed")
            arrays.append(np.asarray(variable[:]))
    return arrays[0], arrays[1]


def _tile_index(title: str) -> tuple[int, int]:
    match = _TILE.search(title)
    if match is None:
        raise RuntimeError(f"Cannot read the sinusoidal tile from granule {title!r}")
    return int(match.group(1)), int(match.group(2))


def _target_grid(bbox: list[float], resolution: float) -> tuple[Any, int, int]:
    """The EPSG:4326 output grid: ``bbox`` widened to multiples of ``resolution``."""
    from rasterio.transform import from_origin

    west = math.floor(bbox[0] / resolution) * resolution
    south = math.floor(bbox[1] / resolution) * resolution
    east = math.ceil(bbox[2] / resolution) * resolution
    north = math.ceil(bbox[3] / resolution) * resolution
    width = round((east - west) / resolution)
    height = round((north - south) / resolution)
    return from_origin(west, north, resolution, resolution), width, height


class VIIRSVegetationIndexMonthlyPlugin(BaseDatasetPlugin):
    """Monthly mean NDVI or EVI from VIIRS 16-day composites.

    Template params:
        index: ``ndvi`` or ``evi``.
        products: which of ``VNP13A1``, ``VJ113A1`` and ``VJ213A1`` to use; all by default.
        resolution: output grid spacing in degrees (default 0.005).
    """

    max_concurrency = 1

    def __init__(
        self,
        index: str,
        products: list[str] | None = None,
        resolution: float = 0.005,
        **_: Any,
    ) -> None:
        if index not in _FIELDS:
            raise ValueError(f"{type(self).__name__}: unsupported index {index!r}; expected one of {list(_FIELDS)}")
        products = list(products or _PRODUCTS)
        unknown = set(products) - set(_PRODUCTS)
        if unknown:
            raise ValueError(f"{type(self).__name__}: unsupported products {sorted(unknown)}; expected {_PRODUCTS}")
        self.index = index
        self.products = products
        self.resolution = float(resolution)

    async def periods(self, start: str, end: str) -> list[str]:
        cutoff = datetime.now(UTC).date() - _PUBLICATION_LAG
        # The last complete month is the latest one ending on or before the cutoff.
        last = cutoff if (cutoff + timedelta(days=1)).day == 1 else cutoff.replace(day=1) - timedelta(days=1)
        return monthly_period_ids(start, end, cutoff=last)

    def _granules(self, year: int, month: int, bbox: list[float]) -> list[dict]:
        _, last_day = calendar.monthrange(year, month)
        # Composites start up to 15 days before their midpoint.
        window = (f"{date(year, month, 1) - timedelta(days=16)}T00:00:00Z", f"{date(year, month, last_day)}T23:59:59Z")
        found = [granule for product in self.products for granule in cmr_granules(product, window, bbox=bbox)]
        in_month = [
            granule for granule in found if (_midpoint(granule).year, _midpoint(granule).month) == (year, month)
        ]
        return _latest_granules(in_month)

    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rioxarray  # noqa: F401  # pyright: ignore[reportUnusedImport]  # activates the .rio accessor
        from rasterio.enums import Resampling
        from rasterio.transform import from_origin

        year, month = int(period_id[:4]), int(period_id[5:7])
        granules = self._granules(year, month, bbox)
        if not granules:
            raise RuntimeError(f"No {'/'.join(self.products)} composites found for {period_id} over {bbox}")

        composites = len({tuple(granule["title"].split(".")[:2]) for granule in granules})
        transform, width, height = _target_grid(bbox, self.resolution)
        total = np.zeros((height, width), dtype="float64")
        count = np.zeros((height, width), dtype="int32")
        session = earthdata_session()
        with tempfile.TemporaryDirectory() as tmp:
            for granule in granules:
                links = data_links(granule, ".h5")
                if not links:
                    raise RuntimeError(f"Granule {granule['title']} has no HDF5 data link")
                path = download(links[0], Path(tmp) / Path(links[0]).name, session=session)
                values, reliability = _read_tile(path, _FIELDS[self.index])
                path.unlink()

                good = (
                    (values >= _VALID_RANGE[0])
                    & (values <= _VALID_RANGE[1])
                    & (reliability >= _KEPT_RANKS[0])
                    & (reliability <= _KEPT_RANKS[1])
                )
                scaled = np.where(good, values.astype("float32") * np.float32(_SCALE), np.float32(np.nan))
                h, v = _tile_index(granule["title"])
                rows, cols = scaled.shape
                tile = xr.DataArray(scaled, dims=("y", "x"))
                tile = tile.rio.write_crs(_SINUSOIDAL).rio.write_transform(
                    from_origin(
                        _GRID_WEST + h * _TILE_SIZE, _GRID_NORTH - v * _TILE_SIZE, _TILE_SIZE / cols, _TILE_SIZE / rows
                    )
                )
                tile = tile.rio.write_nodata(np.nan)
                warped = tile.rio.reproject(
                    "EPSG:4326",
                    transform=transform,
                    shape=(height, width),
                    resampling=Resampling.nearest,
                    nodata=np.nan,
                ).values
                seen = ~np.isnan(warped)
                total[seen] += warped[seen]
                count[seen] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / count, np.nan).astype("float32")
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        monthly = xr.DataArray(
            mean,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "1", "long_name": self.index.upper(), "composites": composites},
        )
        ds = monthly.to_dataset(name=self.index).rio.write_crs("EPSG:4326")
        return cast(xr.Dataset, ds.expand_dims({"t": [np.datetime64(f"{period_id}-01", "ns")]}))
