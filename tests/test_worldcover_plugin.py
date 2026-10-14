"""ESA WorldCover: tile selection, the lazy mosaic, and fill.

`fetch_period` runs for real against GeoTIFF tiles laid out like the bucket's (3 degree tiles,
pixel edges on whole degrees, 0 as no data), at a coarser pixel size so the tiles stay small.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import worldcover

PER_DEGREE = 120  # the real grid is 12000 per degree


def _write_tile(directory: Path, lat: int, lon: int, value: int) -> str:
    name = worldcover._tile_name(lat, lon)
    size = 3 * PER_DEGREE
    with rasterio.open(
        directory / f"{name}.tif",
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:4326",
        transform=from_origin(lon, lat + 3, 1 / PER_DEGREE, 1 / PER_DEGREE),
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(np.full((size, size), value, dtype="uint8"), 1)
    return name


@pytest.fixture
def bucket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(worldcover, "_RES", 1 / PER_DEGREE)
    monkeypatch.setattr(worldcover, "_tile_path", lambda name: str(tmp_path / f"{name}.tif"))
    return tmp_path


def _use_tiles(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    monkeypatch.setattr(worldcover, "_available_tiles", lambda: frozenset(names))


def test_tile_names_and_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    assert worldcover._tile_name(-3, 27) == "S03E027"
    assert worldcover._tile_name(3, -3) == "N03W003"
    _use_tiles(monkeypatch, ["S03E027", "S03E030"])
    assert worldcover._tiles_for([28.8, -2.9, 30.9, -1.0]) == ["S03E027", "S03E030"]
    assert worldcover._tiles_for([28.0, -2.9, 29.0, -1.0]) == ["S03E027"]


def test_tiles_meet_without_a_gap_and_ocean_is_no_data(bucket: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiles = [_write_tile(bucket, -3, 27, 40), _write_tile(bucket, -3, 30, 10)]
    _use_tiles(monkeypatch, tiles)  # the tiles north, at 0 N, are ocean: absent from the bucket

    ds = worldcover.WorldCoverPlugin().fetch_period("2021", [29.5, -1.0, 30.5, 0.5])
    classes = ds["landcover"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert classes.dtype == np.uint8 and classes.encoding["_FillValue"] == 0
    assert hasattr(classes.data, "dask")  # read lazily, chunk by chunk, as the store is written
    values = classes.values
    land = classes.sel(y=slice(-0.0001, -1.0))
    assert (land.sel(x=slice(29.5, 29.9999)) == 40).all()  # west of the seam
    assert (land.sel(x=slice(30.0, 30.5)) == 10).all()  # east of the seam
    assert (classes.sel(y=slice(0.5, 0.0001)) == 0).all()  # ocean
    assert set(np.unique(values)) == {0, 10, 40}


def test_the_output_is_on_whole_pixels_of_the_product_grid(bucket: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [_write_tile(bucket, -3, 27, 30)])

    classes = worldcover.WorldCoverPlugin().fetch_period("2021", [28.001, -2.499, 28.5, -2.0])["landcover"]

    assert np.allclose(classes.x.values * PER_DEGREE - 0.5, np.round(classes.x.values * PER_DEGREE - 0.5))
    assert np.allclose(classes.y.values * PER_DEGREE - 0.5, np.round(classes.y.values * PER_DEGREE - 0.5))
    # The first pixel is the one containing the extent's west edge.
    first = float(classes.x.min())
    assert first - 0.5 / PER_DEGREE <= 28.001 < first + 0.5 / PER_DEGREE
    assert classes.sizes["x"] == 60 and classes.sizes["y"] == 60
    assert (classes == 30).all()


def test_an_extent_with_no_tiles_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_tiles(monkeypatch, [])
    with pytest.raises(RuntimeError, match="No ESA WorldCover tiles"):
        worldcover.WorldCoverPlugin().fetch_period("2021", [-30.0, 0.0, -29.0, 1.0])
