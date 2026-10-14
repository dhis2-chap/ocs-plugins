"""VIIRS vegetation indices: composite selection, quality masking and georeferencing.

`fetch_period` runs for real against HDF5 granules written with the VNP13A1 v002 layout
(the user guide's field names and sinusoidal tile grid). Only CMR and the download are faked.
"""

from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest
from pyproj import Transformer

from ocs_plugins.rasters import viirs_vi

SIZE = 240  # pixels per tile side; real granules have 2400, the reader takes it from the file
H, V = 20, 9  # the sinusoidal tile over most of Rwanda
BBOX = [29.0, -2.5, 30.0, -1.5]


def _granule(title: str, start: str, end: str) -> dict[str, Any]:
    return {
        "title": title,
        "time_start": f"{start}T00:00:00.000Z",
        "time_end": f"{end}T23:59:59.000Z",
        "links": [{"href": f"https://data.example/{title}/{title}.h5"}],
    }


def _write(path: Path, values: np.ndarray, reliability: np.ndarray) -> None:
    with netCDF4.Dataset(path, "w") as nc:
        fields = nc.createGroup("HDFEOS").createGroup("GRIDS").createGroup("VNP_Grid_16Day_VI_500m")
        fields = fields.createGroup("Data Fields")
        fields.createDimension("YDim", SIZE)
        fields.createDimension("XDim", SIZE)
        for name, data, dtype in (
            ("500 m 16 days NDVI", values, "i2"),
            ("500 m 16 days EVI", values, "i2"),
            ("500 m 16 days pixel reliability", reliability, "i1"),
        ):
            fields.createVariable(name, dtype, ("YDim", "XDim"))[:] = data


class Archive:
    """What CMR lists and what the download writes: granule entries and their arrays."""

    def __init__(self) -> None:
        self.granules: list[dict[str, Any]] = []
        self.files: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(self, title: str, start: str, end: str, values: np.ndarray, reliability: np.ndarray) -> None:
        self.granules.append(_granule(title, start, end))
        self.files[title] = (values, reliability)


@pytest.fixture
def archive(monkeypatch: pytest.MonkeyPatch) -> Archive:
    archive = Archive()

    def cmr(short_name: str, temporal: tuple[str, str], bbox: list[float] | None = None, **_: Any) -> list[dict]:
        return [g for g in archive.granules if g["title"].startswith(short_name + ".")]

    def fetch(url: str, target: Path, session: Any = None) -> Path:
        _write(target, *archive.files[url.rsplit("/", 1)[-1].removesuffix(".h5")])
        return target

    monkeypatch.setattr(viirs_vi, "cmr_granules", cmr)
    monkeypatch.setattr(viirs_vi, "download", fetch)
    monkeypatch.setattr(viirs_vi, "earthdata_session", lambda: None)
    return archive


def _uniform(value: int, rank: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return np.full((SIZE, SIZE), value, dtype="int16"), np.full((SIZE, SIZE), rank, dtype="int8")


def test_monthly_mean_of_the_composites_centred_in_the_month(archive: Archive) -> None:
    # Two June composites (midpoints 8 and 24 June), one centred in May that overlaps June.
    archive.add(f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025170000000", "2025-06-01", "2025-06-16", *_uniform(4000))
    archive.add(f"VJ113A1.A2025169.h{H}v{V:02d}.002.2025186000000", "2025-06-17", "2025-07-02", *_uniform(6000))
    archive.add(f"VNP13A1.A2025137.h{H}v{V:02d}.002.2025154000000", "2025-05-17", "2025-06-01", *_uniform(9000))

    ds = viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="ndvi", resolution=0.05).fetch_period("2025-06", BBOX)
    ndvi = ds["ndvi"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert ndvi.attrs["composites"] == 2
    assert np.allclose(ndvi.values, 0.5)


def test_cloud_snow_and_fill_are_dropped(archive: Archive) -> None:
    values, reliability = _uniform(3000)
    reliability[:, -12:] = 9  # cloud over the eastern edge of the tile, inside the extent
    archive.add(f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025170000000", "2025-06-01", "2025-06-16", values, reliability)
    archive.add(f"VJ113A1.A2025169.h{H}v{V:02d}.002.2025186000000", "2025-06-17", "2025-07-02", *_uniform(-15000, -4))

    ndvi = viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="ndvi", resolution=0.05).fetch_period("2025-06", BBOX)
    values_out = ndvi["ndvi"].isel(t=0)

    assert np.isnan(values_out.values).any()
    assert np.allclose(values_out.values[~np.isnan(values_out.values)], 0.3)


def test_a_reprocessed_granule_replaces_the_original(archive: Archive) -> None:
    archive.add(f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025170000000", "2025-06-01", "2025-06-16", *_uniform(1000))
    archive.add(f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025300000000", "2025-06-01", "2025-06-16", *_uniform(2000))

    ndvi = viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="ndvi", resolution=0.05).fetch_period("2025-06", BBOX)

    assert np.allclose(ndvi["ndvi"].values, 0.2)


def test_pixels_land_where_the_sinusoidal_grid_puts_them(archive: Archive) -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(0, 10000, size=(SIZE, SIZE), dtype="int16")
    archive.add(
        f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025170000000",
        "2025-06-01",
        "2025-06-16",
        values,
        np.zeros((SIZE, SIZE), dtype="int8"),
    )

    ndvi = viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="ndvi", resolution=0.05).fetch_period("2025-06", BBOX)
    ndvi = ndvi["ndvi"].isel(t=0)

    to_sinusoidal = Transformer.from_crs("EPSG:4326", viirs_vi._SINUSOIDAL, always_xy=True)
    pixel = viirs_vi._TILE_SIZE / SIZE
    checked = 0
    for lon, lat in [(29.525, -1.975), (29.875, -2.225), (29.125, -1.625)]:
        x, y = to_sinusoidal.transform(lon, lat)
        col = int((x - (viirs_vi._GRID_WEST + H * viirs_vi._TILE_SIZE)) // pixel)
        row = int(((viirs_vi._GRID_NORTH - V * viirs_vi._TILE_SIZE) - y) // pixel)
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            continue  # this point falls in the neighbouring tile
        assert float(ndvi.sel(x=lon, y=lat, method="nearest")) == pytest.approx(values[row, col] * 1e-4, abs=1e-6)
        checked += 1
    assert checked >= 2


def test_selected_products_only(archive: Archive) -> None:
    archive.add(f"VNP13A1.A2025153.h{H}v{V:02d}.002.2025170000000", "2025-06-01", "2025-06-16", *_uniform(4000))
    archive.add(f"VJ113A1.A2025169.h{H}v{V:02d}.002.2025186000000", "2025-06-17", "2025-07-02", *_uniform(8000))

    plugin = viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="evi", products=["VJ113A1"], resolution=0.05)

    assert np.allclose(plugin.fetch_period("2025-06", BBOX)["evi"].values, 0.8)


def test_a_month_with_no_composites_fails_clearly(archive: Archive) -> None:
    with pytest.raises(RuntimeError, match="No VNP13A1/VJ113A1/VJ213A1 composites"):
        viirs_vi.VIIRSVegetationIndexMonthlyPlugin(index="ndvi").fetch_period("2025-06", BBOX)
