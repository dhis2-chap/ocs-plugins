"""CHELSA v2.1 monthly temperature: months offered, conversion to Celsius, and the window read.

`fetch_period` runs for real against a GeoTIFF encoded like CHELSA's (uint16 tenths of a kelvin,
65535 as no data, 30 arc-second pixels); only the file location is faked.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import chelsa

RES = 1 / 120


@pytest.fixture
def month_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = np.full((240, 240), 2932, dtype="uint16")  # 293.2 K = 20.05 degC
    raw[:, :12] = 65535  # no data (sea) along the western edge
    path = tmp_path / "CHELSA_tas_07_2021_V.2.1.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=240,
        width=240,
        count=1,
        dtype="uint16",
        nodata=65535,
        crs="EPSG:4326",
        transform=from_origin(28.0, 0.0, RES, RES),
    ) as dst:
        dst.write(raw, 1)
        dst.scales, dst.offsets = (0.1,), (0.0,)
    monkeypatch.setattr(chelsa, "_path", lambda year, month: str(path))
    return path


def test_kelvin_tenths_become_celsius_and_no_data_is_empty(month_file: Path) -> None:
    ds = chelsa.CHELSATemperaturePlugin().fetch_period("2021-07", [28.0, -2.0, 29.0, -1.0])
    temperature = ds["temperature"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326 and temperature.attrs["units"] == "degC"
    assert np.isnan(temperature.isel(x=slice(0, 12))).all()
    assert np.allclose(temperature.isel(x=slice(12, None)), 20.05, atol=1e-4)
    assert ds.t.values[0] == np.datetime64("2021-07-01")


def test_the_window_covers_the_extent_in_whole_pixels(month_file: Path) -> None:
    temperature = chelsa.CHELSATemperaturePlugin().fetch_period("2021-07", [28.501, -1.499, 29.0, -1.0])["temperature"]

    assert temperature.sizes["x"] == 60 and temperature.sizes["y"] == 60
    assert float(temperature.x.min()) - RES / 2 <= 28.501 < float(temperature.x.min()) + RES / 2


def test_months_stop_where_the_series_ends() -> None:
    plugin = chelsa.CHELSATemperaturePlugin()

    assert asyncio.run(plugin.periods("2021-11-01", "2026-12-31")) == ["2021-11", "2021-12"]
    assert asyncio.run(plugin.periods("1975-01", "1980-02")) == ["1980-01", "1980-02"]
