"""Copernicus DEM GLO-90 from the AWS bucket: tile selection, grid and mosaic.

`fetch_period` runs for real against GeoTIFF tiles laid out like the bucket's, so the
grid alignment and the gaps left by ocean tiles are checked on the path an ingest takes.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import copdem90

RES = 1 / 1200


def _write_tile(directory: Path, lat: int, lon: int, value: float, width: int = 1200) -> str:
    """Write a tile as the bucket stores it: pixel-is-point, centres on whole degrees."""
    name = copdem90._tile_name(lat, lon)
    x_res = 1 / width
    transform = from_origin(lon - x_res / 2, lat + 1 + RES / 2, x_res, RES)
    data = np.full((1200, width), value, dtype="float32")
    path = directory / f"{name}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1200,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return name


@pytest.fixture
def local_bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(copdem90, "_tile_path", lambda name: str(tmp_path / f"{name}.tif"))
    return tmp_path


def _use_tiles(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(copdem90, "_available_tiles", lambda: frozenset(names))


def test_tile_names_follow_the_bucket_scheme() -> None:
    assert copdem90._tile_name(59, 10) == "Copernicus_DSM_COG_30_N59_00_E010_00_DEM"
    assert copdem90._tile_name(-2, 29) == "Copernicus_DSM_COG_30_S02_00_E029_00_DEM"
    assert copdem90._tile_name(0, -1) == "Copernicus_DSM_COG_30_N00_00_W001_00_DEM"


def test_only_tiles_the_bucket_holds_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [copdem90._tile_name(-2, 29)])
    assert copdem90._tiles_for([28.5, -2.5, 29.5, -1.5]) == ["Copernicus_DSM_COG_30_S02_00_E029_00_DEM"]


def test_mosaic_is_on_the_native_grid_and_ocean_is_nan(local_bucket: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    land = _write_tile(local_bucket, -2, 29, 1500.0)
    _use_tiles(monkeypatch, [land])  # the tile to the west, at 28 E, is ocean: absent from the bucket

    ds = copdem90.CopDEM90Plugin().fetch_period("2021", [28.5, -1.5, 29.5, -1.25])
    elevation = ds["elevation"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    # Pixel centres sit on multiples of 3 arc-seconds, as in the tiles themselves.
    assert np.allclose(elevation.x.values * 1200, np.round(elevation.x.values * 1200))
    assert np.allclose(elevation.y.values * 1200, np.round(elevation.y.values * 1200))
    assert float(elevation.x[0]) == pytest.approx(28.5)
    assert float(elevation.y[0]) == pytest.approx(-1.25)
    assert np.isnan(elevation.sel(x=28.75, method="nearest")).all()
    assert (elevation.sel(x=29.25, method="nearest") == 1500.0).all()


def test_wide_pixel_tiles_are_resampled_onto_the_common_grid(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = _write_tile(local_bucket, 59, 10, 300.0, width=800)
    _use_tiles(monkeypatch, [tile])

    elevation = copdem90.CopDEM90Plugin().fetch_period("2021", [10.2, 59.2, 10.4, 59.4])["elevation"]

    assert elevation.sizes["x"] == elevation.sizes["y"] == 241
    assert np.allclose(elevation.values, 300.0)


def test_an_extent_with_no_tiles_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [])
    with pytest.raises(RuntimeError, match="No Copernicus DEM GLO-90 tiles"):
        copdem90.CopDEM90Plugin().fetch_period("2021", [-30.5, 0.5, -30.0, 1.0])
