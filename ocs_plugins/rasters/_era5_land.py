"""ERA5-Land helpers for the CDS plugins in this repo.

Copied from OCS core's ``plugins/rasters/era5_land.py`` as it was before core split that module
up (open-climate-service #505, CLIM-1418), which removed them. Kept here rather than imported from
core's remaining modules, whose helpers are private and can change without notice.
"""

from __future__ import annotations

from datetime import date, datetime

import xarray as xr
from ecmwf.datastores import Client as _CdsClient

# ERA5-Land's grid step is 0.1 degrees; CDS areas are padded by half of it.
_HALF_GRID_STEP = 0.05

_CDS_CATALOGUE_URL = "https://cds.climate.copernicus.eu/api"
_CDS_HOURLY_COLLECTION = "reanalysis-era5-land"
_CDS_DAILY_COLLECTION = "derived-era5-land-daily-statistics"

_GRID_MAPPING_NAMES = ("spatial_ref", "crs")
_EXPVER_DIM = "expver"


def cds_fallback_area(bbox: tuple[float, float, float, float]) -> list[float]:
    """CDS ``area`` (N, W, S, E) for ``bbox``, padded by half a grid step.

    CDS keeps only the grid points inside ``area``, so a bbox edge between two points would drop
    that row or column. The padding stops at the poles and at +-180 degrees, where CDS refuses
    the area.
    """
    xmin, ymin, xmax, ymax = bbox
    return [
        min(ymax + _HALF_GRID_STEP, 90.0),
        max(xmin - _HALF_GRID_STEP, -180.0),
        max(ymin - _HALF_GRID_STEP, -90.0),
        min(xmax + _HALF_GRID_STEP, 180.0),
    ]


def _cds_end_datetime(collection: str) -> datetime:
    """Query the CDS catalogue for the end_datetime of a collection."""
    col = _CdsClient(url=_CDS_CATALOGUE_URL, key="").get_collection(collection)
    if col.end_datetime is None:
        raise RuntimeError(f"CDS collection '{collection}' returned no end_datetime")
    return col.end_datetime


def hourly_availability_cutoff() -> datetime:
    """Return the latest hour for which CDS ERA5-Land hourly data are published."""
    return _cds_end_datetime(_CDS_HOURLY_COLLECTION)


def daily_availability_cutoff() -> date:
    """Return the latest date for which CDS ERA5-Land daily statistics are published."""
    return _cds_end_datetime(_CDS_DAILY_COLLECTION).date()


def collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Merge ERA5's ``expver`` dimension into one series, preferring final over preliminary.

    A CDS request spanning final ERA5 (``expver=1``) and preliminary ERA5T (``expver=5``) returns
    both along an ``expver`` dimension, each NaN where the other has data. Collapsed at download,
    because a reduction applied first turns the all-NaN slice into zeros.
    """
    if _EXPVER_DIM not in ds.dims:
        # Also covers the scalar case, where it is a coordinate rather than a dimension.
        return ds.drop_vars(_EXPVER_DIM, errors="ignore")
    versions = sorted(ds[_EXPVER_DIM].values.tolist())
    merged = ds.sel({_EXPVER_DIM: versions[0]}, drop=True)
    for version in versions[1:]:
        merged = merged.combine_first(ds.sel({_EXPVER_DIM: version}, drop=True))
    return merged.astype({name: ds[name].dtype for name in ds.data_vars})


def drop_auxiliary_variables(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Keep ``variable``, the dimension coordinates and the CRS, dropping everything else.

    ERA5 attaches scalar coordinates (``number``, sometimes ``expver``) that otherwise persist as
    phantom data variables. The grid mapping is kept, since that is where the CRS lives.
    """
    keep = {variable, *ds.dims, *_GRID_MAPPING_NAMES}
    declared = ds[variable].attrs.get("grid_mapping") if variable in ds else None
    if isinstance(declared, str) and declared:
        keep.add(declared)
    extras = [v for v in ds.variables if v not in keep]
    return ds.drop_vars(extras, errors="ignore")
