"""CLMS GPP and NPP: final-version choice, scaling and flags.

`fetch_period` runs for real against GeoTIFFs on the products' 1/336 degree grid, carrying
their scale and offset as the real files do; only the STAC search and S3 location are faked.
"""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import _clms, clms_productivity

RES = 1 / 336
HALF = RES / 2
SIZE = 3 * 336  # a 3x3 degree piece over Rwanda, 28-31 E, 3 S-0
BBOX = [29.0, -2.5, 30.0, -1.5]


def _write(path: Path, raw: np.ndarray, *, scale: float, offset: float, nodata: float) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=1,
        dtype=raw.dtype.name,
        nodata=nodata,
        crs="EPSG:4326",
        transform=from_origin(28 - HALF, 0 + HALF, RES, RES),
    ) as dst:
        dst.write(raw, 1)
        dst.scales, dst.offsets = (scale,), (offset,)
    return path


def _item(item_id: str, start: str, assets: dict[str, Any]) -> dict[str, Any]:
    return {"id": item_id, "properties": {"start_datetime": f"{start}T00:00:00.000000Z"}, "assets": assets}


class Catalogue:
    """The STAC search: returns the items whose start date the query asks for, newest first."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __call__(self, collection: str, **params: Any) -> list[dict[str, Any]]:
        found = self.items
        if "datetime" in params:
            day = params["datetime"][:10]
            found = [item for item in found if item["properties"]["start_datetime"][:10] == day]
        return sorted(found, key=lambda item: item["properties"]["start_datetime"], reverse=True)


def _rt(rt: int, start: str) -> dict[str, Any]:
    return _item(f"c_gls_GPP300-RT{rt}_{start.replace('-', '')}0000_GLOBE_OLCI_V2.0.1_cog", start, {"rt": rt})


@pytest.fixture
def productivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[int, Path]:
    """One file per RT version of the 2026-06-01 dekad, each with its own value."""
    files = {}
    for rt, value in ((0, 1000), (2, 2000), (6, 9000)):  # 0.001 * 9000 = 9 g m-2 day-1
        raw = np.full((SIZE, SIZE), value, dtype="int16")
        raw[:, :10] = -2  # sea
        raw[0, :] = -1  # missing
        files[rt] = _write(tmp_path / f"gpp_rt{rt}.tiff", raw, scale=0.001, offset=0.0, nodata=-1)
    items = [_rt(rt, "2026-06-01") for rt in (0, 2, 6)] + [_rt(0, "2026-08-21"), _rt(1, "2026-08-11")]
    monkeypatch.setattr(_clms, "search", Catalogue(items))
    monkeypatch.setattr(_clms, "asset_path", lambda item, variable: str(files[item["assets"]["rt"]]))
    return files


def test_only_the_final_version_of_a_dekad_is_read(productivity: dict[int, Path]) -> None:
    gpp = clms_productivity.CLMSProductivityDekadalPlugin(product="gpp").fetch_period("2026-06-01", BBOX)["gpp"]

    assert np.allclose(gpp.values, 9.0)
    assert gpp.attrs["units"] == "g m-2 day-1"


def test_sea_and_missing_are_empty(productivity: dict[int, Path]) -> None:
    gpp = (
        clms_productivity.CLMSProductivityDekadalPlugin(product="gpp")
        .fetch_period("2026-06-01", [28.0, -3.0, 29.0, 0.0])["gpp"]
        .isel(t=0)
    )

    assert np.isnan(gpp.isel(x=slice(0, 10))).all()
    assert np.isnan(gpp.isel(y=0)).all()
    assert np.allclose(gpp.isel(y=slice(1, None), x=slice(10, None)), 9.0)


def test_periods_stop_at_the_latest_final_dekad(productivity: dict[int, Path]) -> None:
    plugin = clms_productivity.CLMSProductivityDekadalPlugin(product="npp")

    periods = asyncio.run(plugin.periods("2026-05-01", "2026-12-31"))

    assert periods[-1] == "2026-06-01"  # the later dekads have only preliminary versions


def test_a_dekad_without_a_final_version_is_not_stored(productivity: dict[int, Path]) -> None:
    with pytest.raises(RuntimeError, match="No final"):
        clms_productivity.CLMSProductivityDekadalPlugin(product="gpp").fetch_period("2026-08-21", BBOX)


def test_unknown_product_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported product"):
        clms_productivity.CLMSProductivityDekadalPlugin(product="dmp")
