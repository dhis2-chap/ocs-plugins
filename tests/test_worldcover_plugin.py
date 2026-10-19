"""ESA WorldCover: tile selection, the lazy mosaic, and fill.

`fetch_period` runs for real against GeoTIFF tiles laid out like the bucket's (3 degree tiles,
pixel edges on whole degrees, 0 as no data), at a coarser pixel size so the tiles stay small.
"""

import io
import json
from pathlib import Path
from typing import Any

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
    assert "_FillValue" not in classes.encoding
    assert hasattr(classes.data, "dask")  # read lazily, chunk by chunk, as the store is written
    values = classes.values
    land = classes.sel(y=slice(-0.0001, -1.0))
    assert (land.sel(x=slice(29.5, 29.9999)) == 40).all()  # west of the seam
    assert (land.sel(x=slice(30.0, 30.5)) == 10).all()  # east of the seam
    assert classes.sel(y=slice(0.5, 0.0001)).isnull().all()  # ocean is missing, not a code
    assert set(np.unique(values[np.isfinite(values)])) == {10, 40}


def test_a_mosaic_of_several_tiles_writes_to_zarr(
    bucket: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tiles that do not line up with the chunks still give the uniform chunks Zarr requires."""
    monkeypatch.setattr(worldcover, "_CHUNK", 100)
    tiles = [
        _write_tile(bucket, lat, lon, 10 * (i + 1))
        for i, (lat, lon) in enumerate([(-3, 27), (-3, 30), (-6, 27), (-6, 30)])
    ]
    _use_tiles(monkeypatch, tiles)

    ds = worldcover.WorldCoverPlugin().fetch_period("2021", [28.9, -4.1, 30.7, -1.2])
    ds.to_zarr(tmp_path / "store.zarr", mode="w", zarr_format=3)


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


def test_the_native_grid_has_no_extra_border_pixel() -> None:
    # -2.499 / (1 / 12000) is -29988.000000000004 in floating point; it is still a whole pixel edge.
    x, y = worldcover._grid_axes([28.001, -2.499, 28.5, -2.0])

    assert len(x) == 5988 and len(y) == 5988
    assert x[0] - 0.5 / 12000 <= 28.001 < x[0] + 0.5 / 12000  # the pixel containing the west edge
    assert np.isclose(y[-1], -2.499 + 0.5 / 12000)  # the south edge is a pixel edge


def test_the_tile_grid_is_read_from_the_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []
    grid = {"features": [{"properties": {"ll_tile": "S03E027"}}, {"properties": {"ll_tile": "N51W003"}}]}

    def urlopen(url: str, timeout: float = 0) -> Any:
        requested.append(url)
        return io.BytesIO(json.dumps(grid).encode())

    worldcover._available_tiles.cache_clear()
    monkeypatch.setattr(worldcover.urllib.request, "urlopen", urlopen)
    try:
        assert worldcover._available_tiles() == frozenset({"S03E027", "N51W003"})
    finally:
        worldcover._available_tiles.cache_clear()
    assert requested == [f"{worldcover._BUCKET}/v200/2021/esa_worldcover_grid.geojson"]
