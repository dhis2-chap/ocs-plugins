"""Meta Relative Wealth Index: resolving the country's HDX file, and rebuilding the grid.

`fetch_period` runs for real against a small CSV laid out like an HDX country file: points on
cell centres of the zoom-14 Web Mercator grid, unpopulated cells omitted.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pytest

from ocs_plugins.rasters import rwi

_BASE = "https://data.humdata.org/dataset/rwi/resource"
_FILES = ["npl", "ind_pak", "bosnia-and-herzegovina"]


@pytest.fixture(autouse=True)
def _resources(monkeypatch: pytest.MonkeyPatch) -> None:
    resources = [{"url": f"{_BASE}/{name}{rwi._SUFFIX}"} for name in _FILES]
    monkeypatch.setattr(rwi, "_hdx_resources", lambda: resources)


@pytest.mark.parametrize(
    ("code", "override", "expected"),
    [
        ("NPL", None, "npl"),
        ("pak", None, "ind_pak"),  # one token of a shared file
        ("BIH", "bosnia-and-herzegovina_relative_wealth_index.csv", "bosnia-and-herzegovina"),
    ],
)
def test_resolves_the_country_file(code: str, override: str | None, expected: str) -> None:
    url, filename = rwi._resolve_resource(code, override)

    assert filename == f"{expected}{rwi._SUFFIX}"
    assert url == f"{_BASE}/{filename}"


def test_an_unresolvable_country_names_itself_and_the_slug_files() -> None:
    with pytest.raises(ValueError, match=r"'NOR'.*bosnia-and-herzegovina"):
        rwi._resolve_resource("NOR")


def _lonlat(ix: int, iy: int) -> tuple[float, float]:
    """The centre of zoom-14 cell (ix, iy) counted from an origin near Kathmandu, in degrees."""
    x0, y0 = rwi._to_mercator(np.array([85.0]), np.array([27.5]))
    x = float(x0[0]) + ix * rwi._CELL_M
    y = float(y0[0]) + iy * rwi._CELL_M
    return math.degrees(x / 6378137.0), math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


def _serve(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[float, float, float, float]]) -> None:
    csv = "latitude,longitude,rwi,error\n" + "".join(f"{lat:.6f},{lon:.6f},{v},{e}\n" for lon, lat, v, e in rows)
    monkeypatch.setattr(rwi.httpx, "get", lambda *_args, **_kwargs: _Response(csv.encode()))


def test_rebuilds_the_sparse_points_as_a_mercator_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    # Three of the six cells of a 3 x 2 grid are populated.
    _serve(monkeypatch, [(*_lonlat(0, 0), -0.5, 0.1), (*_lonlat(2, 0), 0.5, 0.2), (*_lonlat(1, 1), 1.0, 0.3)])

    ds = rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, 26, 89, 31], country_code="NPL")

    cube = ds["rwi"].isel(t=0)
    assert ds.rio.crs.to_epsg() == 3857
    assert cube.sizes == {"measure": 2, "y": 2, "x": 3}
    assert bool((np.diff(cube.y.values) < 0).all()), "y should descend"
    np.testing.assert_allclose(np.diff(cube.x.values), rwi._CELL_M)
    # The top row is iy=1, the bottom row iy=0; unpopulated cells are NaN, never 0.
    np.testing.assert_array_equal(
        cube.sel(measure="index").values, np.array([[np.nan, 1.0, np.nan], [-0.5, np.nan, 0.5]], dtype="float32")
    )
    assert float(cube.sel(measure="error").values[0, 1]) == pytest.approx(0.3)


def test_points_off_the_lattice_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    lon, lat = _lonlat(1, 0)
    _serve(monkeypatch, [(*_lonlat(0, 0), 0.0, 0.1), (lon + 0.01, lat, 0.0, 0.1)])  # about half a cell east

    with pytest.raises(ValueError, match="do not lie on a zoom-14 Mercator lattice"):
        rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, 26, 89, 31], country_code="NPL")


def test_the_one_release_is_the_2021_period() -> None:
    plugin = rwi.RelativeWealthIndexPlugin()

    assert asyncio.run(plugin.periods("2020", "2022")) == ["2021"]
    assert asyncio.run(plugin.periods("2022", "2023")) == []


def test_a_country_is_required() -> None:
    plugin: Any = rwi.RelativeWealthIndexPlugin()

    with pytest.raises(ValueError, match="RWI needs a country"):
        plugin.fetch_period("2021", [80, 26, 89, 31])
