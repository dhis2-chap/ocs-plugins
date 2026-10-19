"""Monthly MODIS land surface temperature: which composites count, and how much, and which pixels."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pystac_client
import pytest
import xarray as xr

from ocs_plugins.rasters import modis_lst_monthly as lst


def _item(product: str, start: str, end: str, tile: str = "h24v06") -> Any:
    composite = f"A2024{int(start[5:7]) * 30:03d}"
    return SimpleNamespace(
        id=f"{product}.{composite}.{tile}.061.2024999000000",
        properties={"start_datetime": f"{start}T00:00:00Z", "end_datetime": f"{end}T23:59:59Z"},
    )


def test_a_composite_counts_for_its_days_inside_the_month() -> None:
    assert lst._days_in_month(_item("MOD11A2", "2024-06-25", "2024-07-02"), 2024, 7) == 2
    assert lst._days_in_month(_item("MOD11A2", "2024-07-03", "2024-07-10"), 2024, 7) == 8
    assert lst._days_in_month(_item("MOD11A2", "2024-07-27", "2024-08-03"), 2024, 7) == 5
    assert lst._days_in_month(_item("MOD11A2", "2024-08-04", "2024-08-11"), 2024, 7) == 0


def test_only_terra_composites_overlapping_the_month_are_searched(monkeypatch: pytest.MonkeyPatch) -> None:
    found = [
        _item("MOD11A2", "2024-06-25", "2024-07-02"),
        _item("MYD11A2", "2024-06-25", "2024-07-02"),  # Aqua, a 13:30 overpass
        _item("MOD11A2", "2024-06-17", "2024-06-24"),  # within the search window, not the month
    ]
    catalog = SimpleNamespace(search=lambda **_: SimpleNamespace(items=lambda: found))
    monkeypatch.setattr(pystac_client.Client, "open", lambda *_args, **_kwargs: catalog)

    items = lst.MODISLandSurfaceTemperatureMonthlyPlugin(band="day")._search(2024, 7, [80, 26, 89, 31])

    assert [item.id.split(".")[0] for item in items] == ["MOD11A2"]
    assert lst._window(items[0])[0].isoformat() == "2024-06-25"


def _kelvin(celsius: float) -> int:
    return round((celsius + 273.15) / lst._SCALE_K)


def test_the_monthly_mean_weights_composites_and_drops_fill_and_poor_quality() -> None:
    # Three composites over two cells. Cell 0: 10 °C for 2 days, 20 °C for 8 days. Cell 1: fill,
    # then an error above 2 K (QC bits 6-7 = 2), then 30 °C.
    values = xr.DataArray(
        np.array([[_kelvin(10), 0], [_kelvin(20), _kelvin(-50)], [0, _kelvin(30)]], dtype="uint16"), dims=("time", "x")
    )
    qc = xr.DataArray(np.array([[0, 0], [1 << 6, 2 << 6], [0, 0]], dtype="uint16"), dims=("time", "x"))
    weights = xr.DataArray([2, 8, 5], dims="time")

    mean = lst._weighted_mean(values, qc, weights).values

    np.testing.assert_allclose(mean, [(2 * 10 + 8 * 20) / 10, 30.0], atol=0.02)


@pytest.mark.parametrize("band", ["day", "night"])
def test_fetch_weights_each_composite_by_its_days_in_the_month(monkeypatch: pytest.MonkeyPatch, band: str) -> None:
    """Through `fetch_period`, with odc-stac's load replaced by what it returns for real: one time
    step per composite, stamped with its start day, on latitude/longitude axes."""
    import odc.stac

    composites = {  # start, end, temperature in °C
        "2024-06-25": ("2024-07-02", 10.0),  # 2 days in July
        "2024-07-03": ("2024-07-10", 20.0),  # 8 days
        "2024-07-27": ("2024-08-03", 30.0),  # 5 days
    }
    items = [_item("MOD11A2", start, end) for start, (end, _) in composites.items()]
    monkeypatch.setattr(lst.MODISLandSurfaceTemperatureMonthlyPlugin, "_search", lambda *_: items)

    def load(found: list[Any], bands: list[str], **_: Any) -> xr.Dataset:
        lst_band, qc_band = bands
        assert lst_band == f"LST_{band.capitalize()}_1km" and qc_band == f"QC_{band.capitalize()}"
        values = np.array([[[_kelvin(celsius)] * 2] * 2 for _, celsius in composites.values()], dtype="uint16")
        coords = {
            "time": np.array(list(composites), dtype="datetime64[ns]"),
            "latitude": [27.5, 27.4],
            "longitude": [85.0, 85.1],
        }
        dims = ("time", "latitude", "longitude")
        return xr.Dataset({lst_band: (dims, values), qc_band: (dims, np.zeros_like(values))}, coords=coords)

    monkeypatch.setattr(odc.stac, "load", load)

    ds = lst.MODISLandSurfaceTemperatureMonthlyPlugin(band=band).fetch_period("2024-07", [85, 27, 86, 28])

    variable = ds[f"lst_{band}"]
    assert variable.dims == ("t", "y", "x") and variable.attrs["units"] == "degC"
    assert ds.t.values[0] == np.datetime64("2024-07-01")
    assert ds.rio.crs.to_epsg() == 4326
    np.testing.assert_allclose(variable.values, (2 * 10 + 8 * 20 + 5 * 30) / 15, atol=0.02)


def test_a_cell_with_no_good_composite_is_empty() -> None:
    values = xr.DataArray(np.array([[0], [0]], dtype="uint16"), dims=("time", "x"))
    qc = xr.DataArray(np.zeros((2, 1), dtype="uint16"), dims=("time", "x"))

    assert np.isnan(lst._weighted_mean(values, qc, xr.DataArray([8, 8], dims="time")).values).all()


def test_the_record_starts_with_the_first_whole_month() -> None:
    plugin = lst.MODISLandSurfaceTemperatureMonthlyPlugin(band="night")

    assert asyncio.run(plugin.periods("1999-01", "2000-04")) == ["2000-03", "2000-04"]


def test_an_unknown_band_is_refused() -> None:
    with pytest.raises(ValueError, match="band must be 'day' or 'night'"):
        lst.MODISLandSurfaceTemperatureMonthlyPlugin(band="noon")
