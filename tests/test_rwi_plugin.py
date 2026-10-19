"""Meta Relative Wealth Index: finding the country's HDX file, and rebuilding the grid.

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


def _lonlat(ix: int, iy: int) -> tuple[float, float]:
    """The centre of zoom-14 cell (ix, iy), counted east and north from a cell near Kathmandu."""
    x = math.radians(85.0) * rwi._RADIUS + ix * rwi._CELL_M
    y = math.log(math.tan(math.pi / 4 + math.radians(27.5) / 2)) * rwi._RADIUS + iy * rwi._CELL_M
    return math.degrees(x / rwi._RADIUS), math.degrees(2 * math.atan(math.exp(y / rwi._RADIUS)) - math.pi / 2)


class _Response:
    def __init__(self, content: bytes = b"", payload: Any = None) -> None:
        self.content = content
        self._payload = payload

    def raise_for_status(self) -> _Response:
        return self

    def json(self) -> Any:
        return self._payload


def _serve(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[float, float, float]]) -> list[str]:
    """HDX's resource list and every country file holding *rows*; returns the URLs fetched."""
    resources = {"result": {"resources": [{"url": f"{_BASE}/{name}{rwi._SUFFIX}"} for name in _FILES]}}
    csv = "latitude,longitude,rwi,error\n" + "".join(f"{lat:.6f},{lon:.6f},{v},0.1\n" for lon, lat, v in rows)
    fetched: list[str] = []

    def get(url: str, **_: Any) -> _Response:
        fetched.append(url)
        return _Response(payload=resources) if url == rwi._HDX_PACKAGE else _Response(content=csv.encode())

    monkeypatch.setattr(rwi.httpx, "get", get)
    return fetched


def test_rebuilds_the_sparse_points_as_a_mercator_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    # Three of the six cells of a 3 x 2 grid are populated.
    fetched = _serve(monkeypatch, [(*_lonlat(0, 0), -0.5), (*_lonlat(2, 0), 0.5), (*_lonlat(1, 1), 1.0)])

    ds = rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, 26, 89, 31], country_code="NPL")

    assert fetched[-1] == f"{_BASE}/npl{rwi._SUFFIX}"
    assert ds.rio.crs.to_epsg() == 3857
    cube = ds["rwi"].isel(t=0)
    assert cube.sizes == {"y": 2, "x": 3}
    np.testing.assert_allclose(np.diff(cube.x.values), rwi._CELL_M)
    np.testing.assert_allclose(np.diff(cube.y.values), -rwi._CELL_M)
    # The top row is the northern one; unpopulated cells are NaN, never 0.
    np.testing.assert_array_equal(cube.values, np.array([[np.nan, 1.0, np.nan], [-0.5, np.nan, 0.5]], dtype="float32"))


def test_a_cell_overlapping_the_extent_is_kept_though_its_centre_is_outside(monkeypatch: pytest.MonkeyPatch) -> None:
    lon, lat = _lonlat(0, 0)
    _serve(monkeypatch, [(lon, lat, 0.5)])

    ds = rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, lat + 0.005, 89, 31], country_code="NPL")

    assert float(ds["rwi"].isel(t=0).values.ravel()[0]) == 0.5


def test_a_slug_named_file_is_fetched_by_resource_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = _serve(monkeypatch, [(*_lonlat(0, 0), 0.0)])
    filename = f"bosnia-and-herzegovina{rwi._SUFFIX}"

    rwi.RelativeWealthIndexPlugin(resource_filename=filename).fetch_period("2021", [80, 26, 89, 31], country_code="BIH")

    assert fetched[-1] == f"{_BASE}/{filename}"


def test_a_country_without_a_file_names_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, [(*_lonlat(0, 0), 0.0)])

    with pytest.raises(ValueError, match=r"'nor_relative_wealth_index.csv'.*resource_filename"):
        rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, 26, 89, 31], country_code="NOR")


def test_a_country_is_required() -> None:
    with pytest.raises(ValueError, match="extent.country_code"):
        rwi.RelativeWealthIndexPlugin().fetch_period("2021", [80, 26, 89, 31])


def test_the_one_release_is_the_2021_period() -> None:
    plugin = rwi.RelativeWealthIndexPlugin()

    assert asyncio.run(plugin.periods("2020", "2022")) == ["2021"]
    assert asyncio.run(plugin.periods("2022", "2023")) == []
