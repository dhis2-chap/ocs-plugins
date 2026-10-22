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


def _land_cover(codes: np.ndarray, fill: int | None = None) -> xr.Dataset:
    height, width = codes.shape
    lat = -1.9 - (np.arange(height) + 0.5) * RES  # north to south, as delivered
    lon = 29.0 + (np.arange(width) + 0.5) * RES
    ds = xr.Dataset(
        {
            "lccs_class": (("time", "lat", "lon"), codes[np.newaxis].astype("uint8")),
            "crs": ((), np.int32(0)),
        },
        coords={"time": [np.datetime64("2022-01-01")], "lat": lat, "lon": lon},
    )
    if fill is not None:
        ds["lccs_class"].encoding["_FillValue"] = np.uint8(fill)
    return ds


class FakeCds:
    """Stands in for the CDS client: records requests and writes the given land cover."""

    def __init__(self, codes: np.ndarray, *, zipped: bool, fill: int | None = None) -> None:
        self.codes, self.zipped, self.fill, self.requests = codes, zipped, fill, []

    def __call__(self) -> "FakeCds":
        return self

    def submit(self, dataset: str, request: dict[str, Any]) -> "FakeCds":
        self.requests.append((dataset, request))
        return self

    def download(self, target: str) -> None:
        nc = Path(target).with_suffix(".nc")
        _land_cover(self.codes, self.fill).to_netcdf(nc)
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

    classes = esa_cci_lc._load_classes("2022", BBOX)

    assert classes.dims == ("y", "x") and classes.sizes == {"y": 36, "x": 36}
    assert float(classes.y[0]) > float(classes.y[-1])  # north first
    assert (classes.isel(y=0) == 0).all()  # no data
    assert (classes.isel(y=slice(1, None), x=slice(0, 18)) == 10).all()
    assert (classes.isel(y=slice(1, None), x=slice(18, None)) == 50).all()


@pytest.mark.parametrize(("year", "version"), [("2015", "v2_0_7cds"), ("2016", "v2_1_1")])
def test_the_request_names_the_extent_and_the_years_version(
    monkeypatch: pytest.MonkeyPatch, year: str, version: str
) -> None:
    cds = FakeCds(_codes(), zipped=True)
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", cds)

    esa_cci_lc.ESACCILandCoverFractionPlugin().fetch_period(year, BBOX)

    dataset, request = cds.requests[0]
    assert dataset == "satellite-land-cover"
    assert request["year"] == [year] and request["version"] == [version]
    assert request["area"] == [BBOX[3], BBOX[0], BBOX[1], BBOX[2]]  # north, west, south, east


def test_fractions_share_each_cell_between_the_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", FakeCds(_codes(), zipped=True))

    ds = esa_cci_lc.ESACCILandCoverFractionPlugin(block=6).fetch_period("2022", BBOX)
    fractions = ds["land_cover_fraction"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert ds.t.values[0] == np.datetime64("2022-01-01")
    assert fractions.sizes == {"land_cover_class": 10, "y": 6, "x": 6}
    assert np.allclose(fractions.sel(land_cover_class="cropland").isel(x=slice(0, 3)), 1.0)
    assert np.allclose(fractions.sel(land_cover_class="tree_cover").isel(x=slice(3, None)), 1.0)
    assert np.allclose(fractions.sum("land_cover_class"), 1.0)  # no-data pixels leave the denominator


def test_years_stop_at_the_last_published_map() -> None:
    import asyncio

    years = asyncio.run(esa_cci_lc.ESACCILandCoverFractionPlugin().periods("2020", "2026"))

    assert years == ["2020", "2021", "2022"]


def test_a_declared_fill_value_is_left_out_of_the_fractions(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = np.full((36, 36), 10, dtype="uint8")
    codes[0, :] = 255  # missing, declared as the file's _FillValue
    codes[:6, :6] = 255  # one whole output cell missing
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", FakeCds(codes, zipped=True, fill=255))

    fractions = esa_cci_lc.ESACCILandCoverFractionPlugin(block=6).fetch_period("2022", BBOX)["land_cover_fraction"]
    cropland = fractions.isel(t=0).sel(land_cover_class="cropland")

    assert np.isnan(cropland.isel(y=0, x=0))  # nothing observed in that cell
    assert np.allclose(cropland.isel(y=0, x=slice(1, None)), 1.0)  # 5 of 6 rows observed, all cropland
    assert (esa_cci_lc._load_classes("2022", BBOX).isel(y=0) == 0).all()  # read as no data


@pytest.mark.parametrize("shape", [(1, 36), (36, 1)], ids=["one-row", "one-column"])
def test_an_extent_one_pixel_wide_keeps_both_axes(monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int]) -> None:
    monkeypatch.setattr(esa_cci_lc, "_CdsClient", FakeCds(np.full(shape, 130, dtype="uint8"), zipped=False))

    classes = esa_cci_lc._load_classes("2022", BBOX)

    assert classes.dims == ("y", "x") and classes.shape == shape
    assert (classes == 130).all()
