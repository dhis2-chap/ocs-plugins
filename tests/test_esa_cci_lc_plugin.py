"""ESA CCI / C3S land cover from the CDS: request, file handling, classes and fractions.

`fetch_period` runs for real against netCDF files shaped like the CDS delivers them (``lccs_class``
codes on lat/lon, 1/360 degree pixels with edges on whole multiples); only the CDS client is faked.
"""

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from ocs_plugins.rasters import esa_cci_lc

RES = 1 / 360
BBOX = [29.0, -2.0, 29.1, -1.9]  # 36 x 36 native pixels


def _land_cover(codes: np.ndarray) -> xr.Dataset:
    height, width = codes.shape
    lat = -1.9 - (np.arange(height) + 0.5) * RES  # north to south, as delivered
    lon = 29.0 + (np.arange(width) + 0.5) * RES
    return xr.Dataset(
        {
            "lccs_class": (("time", "lat", "lon"), codes[np.newaxis].astype("uint8")),
            "crs": ((), np.int32(0)),
        },
        coords={"time": [np.datetime64("2022-01-01")], "lat": lat, "lon": lon},
    )


class FakeCds:
    """Stands in for the CDS client: records requests and writes the given land cover."""

    def __init__(self, codes: np.ndarray, *, zipped: bool) -> None:
        self.codes, self.zipped, self.requests = codes, zipped, []

    def __call__(self) -> "FakeCds":
        return self

    def submit(self, dataset: str, request: dict[str, Any]) -> "FakeCds":
        self.requests.append((dataset, request))
        return self

    def download(self, target: str) -> None:
        nc = Path(target).with_suffix(".nc")
        _land_cover(self.codes).to_netcdf(nc)
        if self.zipped:
            with zipfile.ZipFile(target, "w") as archive:
                archive.write(nc, "C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.nc")
        else:
            nc.rename(target)


def _codes() -> np.ndarray:
    codes = np.full((36, 36), 10, dtype="uint8")  # rainfed cropland
    codes[:, 18:] = 50  # tree cover in the eastern half
    codes[0, :] = 0  # no data along the top row
    return codes


@pytest.mark.parametrize("zipped", [True, False], ids=["zip", "netcdf"])
def test_classes_are_read_on_the_native_grid(monkeypatch: pytest.MonkeyPatch, zipped: bool) -> None:
    cds = FakeCds(_codes(), zipped=zipped)
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", cds)

    ds = esa_cci_lc.ESACCILandCoverClassPlugin().fetch_period("2022", BBOX)
    classes = ds["lccs_class"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert classes.dims == ("y", "x") and classes.sizes == {"y": 36, "x": 36}
    assert float(classes.y[0]) > float(classes.y[-1])  # north first
    assert np.isnan(classes.isel(y=0)).all()
    assert (classes.isel(y=slice(1, None), x=slice(0, 18)) == 10).all()
    assert (classes.isel(y=slice(1, None), x=slice(18, None)) == 50).all()
    assert ds.t.values[0] == np.datetime64("2022-01-01")


@pytest.mark.parametrize(("year", "version"), [("2015", "v2_0_7cds"), ("2016", "v2_1_1")])
def test_the_request_names_the_extent_and_the_years_version(
    monkeypatch: pytest.MonkeyPatch, year: str, version: str
) -> None:
    cds = FakeCds(_codes(), zipped=True)
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", cds)

    esa_cci_lc.ESACCILandCoverClassPlugin().fetch_period(year, BBOX)

    dataset, request = cds.requests[0]
    assert dataset == "satellite-land-cover"
    assert request["year"] == [year] and request["version"] == [version]
    assert request["area"] == [BBOX[3], BBOX[0], BBOX[1], BBOX[2]]  # north, west, south, east


def test_fractions_share_each_cell_between_the_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", FakeCds(_codes(), zipped=True))

    fractions = esa_cci_lc.ESACCILandCoverFractionPlugin(block=6).fetch_period("2022", BBOX)["land_cover_fraction"]
    fractions = fractions.isel(t=0)

    assert fractions.sizes == {"land_cover_class": 10, "y": 6, "x": 6}
    assert np.allclose(fractions.sel(land_cover_class="cropland").isel(x=slice(0, 3)), 1.0)
    assert np.allclose(fractions.sel(land_cover_class="tree_cover").isel(x=slice(3, None)), 1.0)
    assert np.allclose(fractions.sum("land_cover_class"), 1.0)  # no-data pixels leave the denominator


def test_years_stop_at_the_last_published_map() -> None:
    import asyncio

    years = asyncio.run(esa_cci_lc.ESACCILandCoverClassPlugin().periods("2020", "2026"))

    assert years == ["2020", "2021", "2022"]
