"""MODIS daily daytime LST: quality masking, conversion to Celsius, mosaic and the days offered.

`fetch_period` runs for real against HDF4 files written with pyhdf in the MOD11A1 layout
(``LST_Day_1km`` in units of 0.02 K with 0 as fill, ``QC_Day`` bitfield); only CMR and the
Earthdata download are faked.
"""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyhdf.SD import SD, SDC

from ocs_plugins.rasters import _modis, modis_lst

SIZE = 120  # pixels per tile side; real tiles have 1200
V = 9
BBOX = [29.5, -2.5, 30.5, -1.5]  # straddles the h20/h21 tile boundary
DAY = "2024-06-15"
# 300.15 K = 27.0 degC, as stored (units of 0.02 K)
WARM = round(300.15 / 0.02)


def _write(path: Path, lst: np.ndarray, qc: np.ndarray) -> None:
    hdf = SD(str(path), SDC.WRITE | SDC.CREATE)
    for name, data, kind in (("LST_Day_1km", lst, SDC.UINT16), ("QC_Day", qc, SDC.UINT8)):
        field = hdf.create(name, kind, data.shape)
        field[:] = data
        field.endaccess()
    hdf.end()


@pytest.fixture
def archive(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    files: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def cmr(short_name: str, temporal: tuple[str, str], bbox: list[float] | None = None, **params: Any) -> list[dict]:
        assert short_name == "MOD11A1" and params.get("version") == "061"
        return [
            {"title": t, "time_start": f"{DAY}T00:00:00.000Z", "links": [{"href": f"https://data.example/{t}.hdf"}]}
            for t in files
        ]

    def fetch(url: str, target: Path, session: Any = None) -> Path:
        _write(target, *files[url.rsplit("/", 1)[-1].removesuffix(".hdf")])
        return target

    for module in (_modis, modis_lst):
        monkeypatch.setattr(module, "cmr_granules", cmr)
    monkeypatch.setattr(_modis, "download", fetch)
    monkeypatch.setattr(_modis, "earthdata_session", lambda: None)
    return files


def _title(h: int) -> str:
    return f"MOD11A1.A2024167.h{h:02d}v{V:02d}.061.2024169031512"


def _tile(raw: int, qc: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((SIZE, SIZE), raw, dtype="uint16"), np.full((SIZE, SIZE), qc, dtype="uint8")


def test_quality_flags() -> None:
    qc = np.array([0b00000000, 0b01000000, 0b10000000, 0b00000001, 0b00000010, 0b11000000], dtype="uint8")
    # good, error <= 1 K, error <= 2 K are kept; error > 2 K, "other quality" and "not produced" are not
    assert modis_lst.good_quality(qc).tolist() == [True, True, False, False, False, False]


def test_good_cells_become_celsius_and_the_rest_are_empty(archive: dict) -> None:
    lst, qc = _tile(WARM, 0)
    # 57 degC flagged as other quality along h20's eastern edge, which lies inside the extent
    lst[:, -6:] = round(330.15 / 0.02)
    qc[:, -6:] = 0b00000001
    archive[_title(20)] = (lst, qc)
    archive[_title(21)] = _tile(WARM, 0b01000000)

    ds = modis_lst.MODISLSTDayPlugin(resolution=0.05).fetch_period(DAY, BBOX)
    lst_day = ds["lst_day"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326 and lst_day.attrs["units"] == "degC"
    assert ds.t.values[0] == np.datetime64(DAY)
    values = lst_day.values[~np.isnan(lst_day.values)]
    assert values.size and np.allclose(values, 27.0, atol=0.01)  # the 57 degC cells were masked
    assert np.isnan(lst_day.values).any()
    assert not np.isnan(lst_day.isel(x=-1)).all()  # the h21 tile reaches the output too


def test_days_stop_at_the_latest_published_granule(archive: dict) -> None:
    archive[_title(20)] = _tile(WARM, 0)

    days = asyncio.run(modis_lst.MODISLSTDayPlugin().periods("2024-06-13", "2026-01-01"))

    assert days == ["2024-06-13", "2024-06-14", "2024-06-15"]


def test_a_day_without_granules_fails_clearly(archive: dict) -> None:
    with pytest.raises(RuntimeError, match="No MOD11A1"):
        modis_lst.MODISLSTDayPlugin().fetch_period(DAY, BBOX)
