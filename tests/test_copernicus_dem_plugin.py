"""Copernicus DEM GLO-30 and GLO-90 from the AWS buckets: tile selection, grid and mosaic.

`fetch_period` runs for real against GeoTIFF tiles laid out like each bucket's, so the grid
alignment and the gaps left by ocean tiles are checked on the path an ingest takes.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import copernicus_dem

# Resolution in metres -> (tile name code, pixels per degree).
GRIDS = {30: ("10", 3600), 90: ("30", 1200)}


def _write_tile(directory: Path, resolution: int, lat: int, lon: int, value: float, width: int | None = None) -> str:
    """Write a tile as the bucket stores it: pixel-is-point, centres on whole multiples of the pixel."""
    code, per_degree = GRIDS[resolution]
    width = width or per_degree
    res, x_res = 1 / per_degree, 1 / width
    name = copernicus_dem._tile_name(code, lat, lon)
    path = directory / f"{name}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=per_degree,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(lon - x_res / 2, lat + 1 + res / 2, x_res, res),
    ) as dst:
        dst.write(np.full((per_degree, width), value, dtype="float32"), 1)
    return name


@pytest.fixture
def local_bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(copernicus_dem, "_tile_path", lambda bucket, name: str(tmp_path / f"{name}.tif"))
    return tmp_path


def _use_tiles(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(copernicus_dem, "_available_tiles", lambda bucket: frozenset(names))


def test_tile_names_follow_each_buckets_scheme() -> None:
    assert copernicus_dem._tile_name("30", 59, 10) == "Copernicus_DSM_COG_30_N59_00_E010_00_DEM"
    assert copernicus_dem._tile_name("10", -2, 29) == "Copernicus_DSM_COG_10_S02_00_E029_00_DEM"
    assert copernicus_dem._tile_name("30", 0, -1) == "Copernicus_DSM_COG_30_N00_00_W001_00_DEM"


@pytest.mark.parametrize("resolution", [30, 90])
def test_only_tiles_the_bucket_holds_are_read(monkeypatch: pytest.MonkeyPatch, resolution: int) -> None:
    code = GRIDS[resolution][0]
    _use_tiles(monkeypatch, [copernicus_dem._tile_name(code, -2, 29)])
    plugin = copernicus_dem.CopernicusDEMPlugin(resolution=resolution)
    assert plugin._tiles_for([28.5, -2.5, 29.5, -1.5]) == [f"Copernicus_DSM_COG_{code}_S02_00_E029_00_DEM"]


@pytest.mark.parametrize("resolution", [30, 90])
def test_mosaic_is_on_the_native_grid_and_ocean_is_nan(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch, resolution: int
) -> None:
    per_degree = GRIDS[resolution][1]
    land = _write_tile(local_bucket, resolution, -2, 29, 1500.0)
    _use_tiles(monkeypatch, [land])  # the tile to the west, at 28 E, is ocean: absent from the bucket

    ds = copernicus_dem.CopernicusDEMPlugin(resolution=resolution).fetch_period("2021", [28.5, -1.5, 29.5, -1.25])
    elevation = ds["elevation"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    # Pixel centres sit on whole multiples of the pixel size, as in the tiles themselves.
    assert np.allclose(elevation.x.values * per_degree, np.round(elevation.x.values * per_degree))
    assert np.allclose(elevation.y.values * per_degree, np.round(elevation.y.values * per_degree))
    assert float(elevation.x[0]) == pytest.approx(28.5)
    assert float(elevation.y[0]) == pytest.approx(-1.25)
    assert elevation.sizes["x"] == per_degree + 1
    assert np.isnan(elevation.sel(x=28.75, method="nearest")).all()
    assert (elevation.sel(x=29.25, method="nearest") == 1500.0).all()


@pytest.mark.parametrize("resolution", [30, 90])
def test_wide_pixel_tiles_are_resampled_onto_the_common_grid(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch, resolution: int
) -> None:
    per_degree = GRIDS[resolution][1]
    tile = _write_tile(local_bucket, resolution, 59, 10, 300.0, width=per_degree * 2 // 3)
    _use_tiles(monkeypatch, [tile])

    elevation = copernicus_dem.CopernicusDEMPlugin(resolution=resolution).fetch_period(
        "2021", [10.2, 59.2, 10.4, 59.4]
    )["elevation"]

    assert elevation.sizes["x"] == elevation.sizes["y"] == round(0.2 * per_degree) + 1
    assert np.allclose(elevation.values, 300.0)


def test_an_extent_with_no_tiles_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [])
    with pytest.raises(RuntimeError, match="No Copernicus DEM tiles"):
        copernicus_dem.CopernicusDEMPlugin().fetch_period("2021", [-30.5, 0.5, -30.0, 1.0])


def test_an_unknown_resolution_is_refused() -> None:
    with pytest.raises(ValueError, match="resolution must be one of"):
        copernicus_dem.CopernicusDEMPlugin(resolution=10)


@pytest.mark.parametrize("resolution", [30, 90])
def test_an_extent_on_whole_degrees_takes_its_edges_from_the_neighbouring_tiles(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch, resolution: int
) -> None:
    # The pixel centres on 30 E belong to the E030 tile, and those on 2 S to the S03 tile.
    tiles = [
        _write_tile(local_bucket, resolution, lat, lon, value)
        for lat, lon, value in [(-2, 29, 100.0), (-2, 30, 200.0), (-3, 29, 300.0), (-3, 30, 400.0)]
    ]
    _use_tiles(monkeypatch, tiles)

    elevation = copernicus_dem.CopernicusDEMPlugin(resolution=resolution).fetch_period("2021", [29, -2, 30, -1])
    elevation = elevation["elevation"].isel(t=0)

    assert not np.isnan(elevation).any()
    assert float(elevation.sel(x=29.5, y=-1.5, method="nearest")) == 100.0
    assert (elevation.isel(x=-1, y=slice(0, -1)) == 200.0).all()  # the 30 E column
    assert (elevation.isel(y=-1, x=slice(0, -1)) == 300.0).all()  # the 2 S row
    assert float(elevation.isel(x=-1, y=-1)) == 400.0
