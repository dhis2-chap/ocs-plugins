"""Reading downloaded netCDF files from ingest threads.

netCDF4 and the HDF5 library it links are not thread-safe, and ingests fetch in threads:
up to a plugin's ``max_concurrency`` per job, and several jobs at once. Two reads at the
same time can corrupt HDF5's state and crash the whole process with a segmentation fault,
so every read of a downloaded netCDF file goes through ``NETCDF_LOCK``.

The lock only protects anything if the plugins here and OCS core share it, so OCS's own is
used when it has one (dhis2/open-climate-service#465), and this module's until then.
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import xarray as xr

try:
    from open_climate_service.shared.netcdf import NETCDF_LOCK, load_netcdf, load_netcdfs  # type: ignore[import-not-found]
except ImportError:
    NETCDF_LOCK = threading.Lock()

    def load_netcdf(path: str | Path) -> xr.Dataset:
        """Read a netCDF file into memory under the process-wide lock."""
        with NETCDF_LOCK, xr.open_dataset(path, engine="netcdf4") as ds:
            return ds.load()

    def load_netcdfs(paths: Sequence[str | Path], **kwargs: Any) -> xr.Dataset:
        """Read several netCDF files as one dataset under the lock, without dask reading them in parallel."""
        with NETCDF_LOCK, xr.open_mfdataset(paths, engine="netcdf4", **kwargs) as ds:
            return ds.load(scheduler="synchronous")


__all__ = ["NETCDF_LOCK", "load_netcdf", "load_netcdfs"]
