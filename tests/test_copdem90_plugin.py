"""Copernicus DEM GLO-90 from the AWS bucket: tile selection, grid and mosaic.

`fetch` runs for real against GeoTIFF tiles laid out like the bucket's, so the grid
alignment and the gaps left by ocean tiles are checked on the path an ingest takes.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import copdem90

PER_DEGREE = 1200


def _write_tile(directory: Path, lat: int, lon: int, value: float, width: int = PER_DEGREE) -> str:
    """Write a tile as the bucket stores it: pixel-is-point, centres on whole multiples of the pixel."""
    res, x_res = 1 / PER_DEGREE, 1 / width
    name = copdem90._tile_name(lat, lon)
    with rasterio.open(
        directory / f"{name}.tif",
        "w",
        driver="GTiff",
        height=PER_DEGREE,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(lon - x_res / 2, lat + 1 + res / 2, x_res, res),
    ) as dst:
        dst.write(np.full((PER_DEGREE, width), value, dtype="float32"), 1)
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
    plugin = copdem90.CopDEM90Plugin()
    assert plugin._tiles_for([28.5, -2.5, 29.5, -1.5]) == ["Copernicus_DSM_COG_30_S02_00_E029_00_DEM"]


def test_mosaic_is_on_the_native_grid_and_ocean_is_nan(local_bucket: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    land = _write_tile(local_bucket, -2, 29, 1500.0)
    _use_tiles(monkeypatch, [land])  # the tile to the west, at 28 E, is ocean: absent from the bucket

    ds = copdem90.CopDEM90Plugin().fetch([28.5, -1.5, 29.5, -1.25])
    elevation = ds["elevation"]

    assert ds.rio.crs.to_epsg() == 4326
    assert elevation.dims == ("y", "x")  # elevation has no time axis
    # Pixel centres sit on whole multiples of the pixel size, as in the tiles themselves.
    assert np.allclose(elevation.x.values * PER_DEGREE, np.round(elevation.x.values * PER_DEGREE))
    assert np.allclose(elevation.y.values * PER_DEGREE, np.round(elevation.y.values * PER_DEGREE))
    assert float(elevation.x[0]) == pytest.approx(28.5)
    assert float(elevation.y[0]) == pytest.approx(-1.25)
    assert elevation.sizes["x"] == PER_DEGREE + 1
    assert np.isnan(elevation.sel(x=28.75, method="nearest")).all()
    assert (elevation.sel(x=29.25, method="nearest") == 1500.0).all()


def test_wide_pixel_tiles_are_resampled_onto_the_common_grid(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = _write_tile(local_bucket, 59, 10, 300.0, width=PER_DEGREE * 2 // 3)
    _use_tiles(monkeypatch, [tile])

    elevation = copdem90.CopDEM90Plugin().fetch([10.2, 59.2, 10.4, 59.4])["elevation"]

    assert elevation.sizes["x"] == elevation.sizes["y"] == round(0.2 * PER_DEGREE) + 1
    assert np.allclose(elevation.values, 300.0)


def test_an_extent_on_whole_degrees_takes_its_edges_from_the_neighbouring_tiles(
    local_bucket: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pixel centres on 30 E belong to the E030 tile, and those on 2 S to the S03 tile.
    tiles = [
        _write_tile(local_bucket, lat, lon, value)
        for lat, lon, value in [(-2, 29, 100.0), (-2, 30, 200.0), (-3, 29, 300.0), (-3, 30, 400.0)]
    ]
    _use_tiles(monkeypatch, tiles)

    elevation = copdem90.CopDEM90Plugin().fetch([29, -2, 30, -1])["elevation"]

    assert not np.isnan(elevation).any()
    assert float(elevation.sel(x=29.5, y=-1.5, method="nearest")) == 100.0
    assert (elevation.isel(x=-1, y=slice(0, -1)) == 200.0).all()  # the 30 E column
    assert (elevation.isel(y=-1, x=slice(0, -1)) == 300.0).all()  # the 2 S row
    assert float(elevation.isel(x=-1, y=-1)) == 400.0


def test_an_extent_with_no_tiles_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [])
    with pytest.raises(RuntimeError, match="No Copernicus DEM tiles"):
        copdem90.CopDEM90Plugin().fetch([-30.5, 0.5, -30.0, 1.0])
