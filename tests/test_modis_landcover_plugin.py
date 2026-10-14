"""MODIS MCD12Q1 land cover: HDF4 reading, sinusoidal placement and the multi-tile mosaic.

`fetch_period` runs for real against HDF4 files written with pyhdf in the MCD12Q1 layout
(``LC_Type1``, fill 255); only CMR and the Earthdata download are faked.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyhdf.SD import SD, SDC
from pyproj import Transformer

from ocs_plugins.rasters import modis_landcover

SIZE = 240  # pixels per tile side; real tiles have 2400, the reader takes it from the file
V = 9
BBOX = [29.5, -2.5, 30.5, -1.5]  # straddles the h20/h21 tile boundary near 30 E


def _write(path: Path, classes: np.ndarray) -> None:
    hdf = SD(str(path), SDC.WRITE | SDC.CREATE)
    field = hdf.create("LC_Type1", SDC.UINT8, classes.shape)
    field[:] = classes
    field.endaccess()
    hdf.end()


@pytest.fixture
def archive(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Granule title -> class array; CMR lists them and the download writes them."""
    files: dict[str, np.ndarray] = {}

    def cmr(short_name: str, temporal: tuple[str, str], bbox: list[float] | None = None, **_: Any) -> list[dict]:
        return [{"title": title, "links": [{"href": f"https://data.example/{title}.hdf"}]} for title in files]

    def fetch(url: str, target: Path, session: Any = None) -> Path:
        _write(target, files[url.rsplit("/", 1)[-1].removesuffix(".hdf")])
        return target

    monkeypatch.setattr(modis_landcover, "cmr_granules", cmr)
    monkeypatch.setattr(modis_landcover, "download", fetch)
    monkeypatch.setattr(modis_landcover, "earthdata_session", lambda: None)
    return files


def _title(h: int) -> str:
    return f"MCD12Q1.A2024001.h{h:02d}v{V:02d}.061.2025205202704"


def test_tiles_are_mosaicked_onto_one_grid(archive: dict[str, np.ndarray]) -> None:
    archive[_title(20)] = np.full((SIZE, SIZE), 9, dtype="uint8")  # savanna
    archive[_title(21)] = np.full((SIZE, SIZE), 12, dtype="uint8")  # cropland

    ds = modis_landcover.MODISLandCoverPlugin(resolution=0.05).fetch_period("2024", BBOX)
    classes = ds["land_cover"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert not np.isnan(classes).any()  # both tiles reach the output, with no gap between them
    assert set(np.unique(classes.values)) == {9.0, 12.0}
    assert (classes.isel(x=0) == 9).all() and (classes.isel(x=-1) == 12).all()
    assert ds.t.values[0] == np.datetime64("2024-01-01")


def test_fill_is_empty(archive: dict[str, np.ndarray]) -> None:
    tile = np.full((SIZE, SIZE), 17, dtype="uint8")
    tile[: SIZE // 2] = 255
    archive[_title(20)] = tile

    classes = modis_landcover.MODISLandCoverPlugin(resolution=0.05).fetch_period("2024", [29.0, -9.0, 29.5, -1.0])
    values = classes["land_cover"].values

    assert np.isnan(values).any() and set(np.unique(values[~np.isnan(values)])) == {17.0}


def test_pixels_land_where_the_sinusoidal_grid_puts_them(archive: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(0)
    tile = rng.integers(1, 18, size=(SIZE, SIZE), dtype="uint8")
    archive[_title(20)] = tile

    classes = modis_landcover.MODISLandCoverPlugin(resolution=0.05).fetch_period("2024", [29.0, -2.5, 29.8, -1.5])
    classes = classes["land_cover"].isel(t=0)

    to_sinusoidal = Transformer.from_crs("EPSG:4326", modis_landcover._SINUSOIDAL, always_xy=True)
    pixel = modis_landcover._TILE_SIZE / SIZE
    for lon, lat in [(29.125, -1.625), (29.525, -1.975), (29.725, -2.425)]:
        x, y = to_sinusoidal.transform(lon, lat)
        col = int((x - (modis_landcover._GRID_WEST + 20 * modis_landcover._TILE_SIZE)) // pixel)
        row = int(((modis_landcover._GRID_NORTH - V * modis_landcover._TILE_SIZE) - y) // pixel)
        assert float(classes.sel(x=lon, y=lat, method="nearest")) == tile[row, col]


def test_a_year_without_granules_fails_clearly(archive: dict[str, np.ndarray]) -> None:
    with pytest.raises(RuntimeError, match="No MCD12Q1 land cover"):
        modis_landcover.MODISLandCoverPlugin().fetch_period("2024", BBOX)
