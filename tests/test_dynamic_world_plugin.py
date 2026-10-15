"""Dynamic World: months offered, the grid Xee is asked for, a month with no scenes, and credentials.

Earth Engine and Xee are replaced by stand-ins: Earth Engine objects record the calls made on them,
and `xr.open_dataset` records the grid it is asked to read and answers with values that differ by
row and column, so the test checks that the grid is passed on and read back the right way round.
"""

import asyncio
import sys
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from ocs_plugins.rasters import dynamic_world


class _Chain:
    """Any Earth Engine object: every method returns itself, and the scene count is configurable."""

    def __init__(self, scenes: int, calls: list[str]) -> None:
        self._scenes, self._calls = scenes, calls

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> Any:
            self._calls.append(name)
            return self

        return method

    def size(self) -> Any:
        return SimpleNamespace(getInfo=lambda: self._scenes)


def _fake_ee(monkeypatch: pytest.MonkeyPatch, scenes: int = 3) -> tuple[list[dict[str, Any]], list[str]]:
    reads: list[dict[str, Any]] = []
    calls: list[str] = []
    chain = _Chain(scenes, calls)
    ee = SimpleNamespace(
        ImageCollection=lambda _: chain,
        Geometry=SimpleNamespace(Rectangle=lambda _: chain),
        Date=lambda _: chain,
    )
    monkeypatch.setattr(dynamic_world, "_ee", lambda: ee)

    def open_dataset(collection: Any, **kwargs: Any) -> xr.Dataset:
        reads.append(kwargs)
        width, height = kwargs["shape_2d"]
        # Row r, column c holds (r + 2c) % 9, as floats like Xee returns.
        values = (np.arange(height)[:, None] + 2 * np.arange(width)[None, :]) % 9
        label = xr.DataArray(values[None].astype("float32"), dims=("time", "y", "x"))
        return xr.Dataset({"label": label})

    monkeypatch.setattr(dynamic_world.xr, "open_dataset", open_dataset)
    return reads, calls


def test_months_are_complete_months_within_the_record() -> None:
    plugin = dynamic_world.DynamicWorldPlugin()
    months = asyncio.run(plugin.periods("2015-01-01", "2015-09-30"))
    assert months == ["2015-07", "2015-08", "2015-09"]

    this_month = date.today().strftime("%Y-%m")
    recent = asyncio.run(plugin.periods("2026-01-01", "2099-12-31"))
    assert this_month not in recent and recent[-1] < this_month


def test_the_month_is_read_on_the_output_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, calls = _fake_ee(monkeypatch)
    bbox = [30.0, -2.25, 30.25, -2.0]  # 250 x 250 cells at 0.001 degrees

    ds = dynamic_world.DynamicWorldPlugin().fetch_period("2024-03", bbox)

    assert reads == [
        {
            "engine": "ee",
            "crs": "EPSG:4326",
            "crs_transform": (0.001, 0, 30.0, 0, -0.001, -2.0),
            "shape_2d": (250, 250),
        }
    ]
    landcover = ds["landcover"].isel(t=0)
    assert landcover.dtype == np.uint8 and landcover.shape == (250, 250)
    assert int(landcover[1, 0]) == 1 and int(landcover[0, 1]) == 2  # rows are y, columns are x
    assert float(landcover.x[0]) == pytest.approx(30.0005) and float(landcover.y[0]) == pytest.approx(-2.0005)
    assert ds["t"].values[0] == np.datetime64("2024-03-01")
    # Requested at the output grid directly: no 10 m aggregation of the month, which runs out of memory.
    assert "mode" in calls and "reduceResolution" not in calls


def test_a_month_without_scenes_is_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    reads, _ = _fake_ee(monkeypatch, scenes=0)

    ds = dynamic_world.DynamicWorldPlugin().fetch_period("2015-07", [30.0, -2.01, 30.01, -2.0])

    assert not reads
    assert set(np.unique(ds["landcover"].values)) == {255}


def test_a_missing_project_says_what_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EARTHENGINE_PROJECT", raising=False)
    monkeypatch.setattr(dynamic_world, "_initialised", False)

    with pytest.raises(RuntimeError, match="EARTHENGINE_PROJECT"):
        dynamic_world._ee()


def test_a_service_account_key_is_passed_to_earth_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Earth Engine client does not read GOOGLE_APPLICATION_CREDENTIALS itself, so the plugin does."""
    import google.auth

    initialised: list[tuple[Any, ...]] = []
    fake_ee = SimpleNamespace(
        Initialize=lambda credentials, project, opt_url: initialised.append((credentials, project, opt_url))
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(google.auth, "default", lambda scopes: ("service-account", "from-key"))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/keys/instance.json")
    monkeypatch.setenv("EARTHENGINE_PROJECT", "instance-project")
    monkeypatch.setattr(dynamic_world, "_initialised", False)

    dynamic_world._ee()

    assert initialised == [("service-account", "instance-project", "https://earthengine-highvolume.googleapis.com")]
