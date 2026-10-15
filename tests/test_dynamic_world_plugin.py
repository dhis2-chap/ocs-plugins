"""Dynamic World: months offered, the grid and its tiles, and a month with no scenes.

Earth Engine is replaced by a stand-in that records the requests and answers `computePixels` with
each tile's position, so the test checks that every tile lands where it belongs in the grid.
"""

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

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
    requests: list[dict[str, Any]] = []
    calls: list[str] = []
    chain = _Chain(scenes, calls)

    def compute_pixels(request: dict[str, Any]) -> np.ndarray:
        requests.append(request)
        grid = request["grid"]
        width, height = grid["dimensions"]["width"], grid["dimensions"]["height"]
        # Each tile is filled with a value from its column, so a misplaced tile shows.
        column = round((grid["affineTransform"]["translateX"] - 30.0) / 0.001) // dynamic_world._TILE
        values = np.full((height, width), column % 8, dtype="uint8")
        return values.astype([("label_mode", "uint8")]).view(np.recarray)

    ee = SimpleNamespace(
        ImageCollection=lambda _: chain,
        Geometry=SimpleNamespace(Rectangle=lambda _: chain),
        Projection=lambda _: chain,
        Reducer=SimpleNamespace(mode=lambda: chain),
        data=SimpleNamespace(computePixels=compute_pixels),
    )
    monkeypatch.setattr(dynamic_world, "_ee", lambda: ee)
    return requests, calls


def test_months_are_complete_months_within_the_record() -> None:
    plugin = dynamic_world.DynamicWorldPlugin()
    months = asyncio.run(plugin.periods("2015-01-01", "2015-09-30"))
    assert months == ["2015-07", "2015-08", "2015-09"]

    this_month = date.today().strftime("%Y-%m")
    recent = asyncio.run(plugin.periods("2026-01-01", "2099-12-31"))
    assert this_month not in recent and recent[-1] < this_month


def test_tiles_cover_the_grid_once_and_land_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    requests, calls = _fake_ee(monkeypatch)
    monkeypatch.setattr(dynamic_world, "_TILE", 100)
    bbox = [30.0, -2.25, 30.25, -2.0]  # 250 x 250 cells at 0.001 degrees: 3 x 3 tiles

    ds = dynamic_world.DynamicWorldPlugin().fetch_period("2024-03", bbox)

    landcover = ds["landcover"].isel(t=0)
    assert landcover.shape == (250, 250)
    assert len(requests) == 9
    # Columns 0-99 came from tile column 0, 100-199 from 1, 200-249 from 2, in every row.
    assert set(np.unique(landcover.values[:, :100])) == {0}
    assert set(np.unique(landcover.values[:, 100:200])) == {1}
    assert set(np.unique(landcover.values[:, 200:])) == {2}
    assert float(landcover.x[0]) == pytest.approx(30.0005) and float(landcover.y[0]) == pytest.approx(-2.0005)
    assert ds["t"].values[0] == np.datetime64("2024-03-01")
    # Coarser than 10 m, so each cell is the most frequent 10 m class within it.
    assert "reduceResolution" in calls


def test_a_month_without_scenes_is_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    requests, _ = _fake_ee(monkeypatch, scenes=0)

    ds = dynamic_world.DynamicWorldPlugin().fetch_period("2015-07", [30.0, -2.01, 30.01, -2.0])

    assert not requests
    assert set(np.unique(ds["landcover"].values)) == {255}


def test_native_resolution_skips_the_aggregation(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _fake_ee(monkeypatch)

    dynamic_world.DynamicWorldPlugin(pixel_degrees=0.0001).fetch_period("2024-03", [30.0, -2.001, 30.001, -2.0])

    assert "reduceResolution" not in calls


def test_a_missing_project_says_what_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EARTHENGINE_PROJECT", raising=False)
    monkeypatch.setattr(dynamic_world, "_initialised", False)

    with pytest.raises(RuntimeError, match="EARTHENGINE_PROJECT"):
        dynamic_world._ee()


def test_a_service_account_key_is_passed_to_earth_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Earth Engine client does not read GOOGLE_APPLICATION_CREDENTIALS itself, so the plugin does."""
    import sys

    import google.auth

    initialised: list[tuple[Any, str]] = []
    fake_ee = SimpleNamespace(Initialize=lambda credentials, project: initialised.append((credentials, project)))
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(google.auth, "default", lambda scopes: ("service-account", "from-key"))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/keys/instance.json")
    monkeypatch.setenv("EARTHENGINE_PROJECT", "instance-project")
    monkeypatch.setattr(dynamic_world, "_initialised", False)

    dynamic_world._ee()

    assert initialised == [("service-account", "instance-project")]
