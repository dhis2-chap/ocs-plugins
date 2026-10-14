"""CLMS NDVI 300 m: item lookup, window read, scaling and flags.

`fetch_period` runs for real against a GeoTIFF on the product's 1/336 degree grid; only the
STAC search and the S3 location are faked.
"""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ocs_plugins.rasters import clms_ndvi

RES = 1 / 336
HALF = RES / 2
BBOX = [29.0, -2.5, 30.0, -1.5]


def _item(period: str, href: str) -> dict[str, Any]:
    return {
        "id": f"c_gls_NDVI300_{period.replace('-', '')}0000_GLOBE_OLCI_V3.0.1_cog",
        "properties": {"start_datetime": f"{period}T00:00:00.000000Z"},
        "assets": {
            "ndvi": {
                "href": href,
                "alternate": {
                    "https": {
                        "href": "https://download.example/odata/v1/Products(x)/Nodes(c_gls_NDVI300_x_cog)/"
                        "Nodes(c_gls_NDVI300-NDVI_202609010000_GLOBE_OLCI_V3.0.1.tiff)/$value"
                    }
                },
            }
        },
    }


@pytest.fixture
def product(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """A 3x3 degree piece of the product grid over Rwanda; returns its raw values."""
    size = 3 * 336
    raw = np.full((size, size), 200, dtype="uint8")  # NDVI 0.72
    raw[:, :10] = 254  # water along the western edge
    raw[0, :] = 255  # missing along the top row
    path = tmp_path / "ndvi.tiff"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        nodata=255,
        crs="EPSG:4326",
        transform=from_origin(28 - HALF, 0 + HALF, RES, RES),
    ) as dst:
        dst.write(raw, 1)
    monkeypatch.setattr(clms_ndvi, "_search", lambda **_: [_item("2026-09-01", "s3://eodata/x/")])
    monkeypatch.setattr(clms_ndvi, "_ndvi_path", lambda item: str(path))
    return raw


def test_values_are_scaled_and_flags_are_empty(product: np.ndarray) -> None:
    ds = clms_ndvi.CLMSNDVIDekadalPlugin().fetch_period("2026-09-01", [28.0, -3.0, 29.0, 0.0])
    ndvi = ds["ndvi"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert np.isnan(ndvi.isel(x=slice(0, 10))).all()  # water
    assert np.isnan(ndvi.isel(y=0)).all()  # missing
    assert np.allclose(ndvi.isel(y=slice(1, None), x=slice(10, None)), 0.72)


def test_the_window_is_on_the_product_grid(product: np.ndarray) -> None:
    ndvi = clms_ndvi.CLMSNDVIDekadalPlugin().fetch_period("2026-09-01", BBOX)["ndvi"]

    assert np.allclose(ndvi.x.values * 336, np.round(ndvi.x.values * 336))
    assert np.allclose(ndvi.y.values * 336, np.round(ndvi.y.values * 336))
    assert float(ndvi.x.min()) <= BBOX[0] + HALF and float(ndvi.x.max()) >= BBOX[2] - HALF
    assert float(ndvi.y.min()) <= BBOX[1] + HALF and float(ndvi.y.max()) >= BBOX[3] - HALF
    assert ndvi.sizes["x"] == ndvi.sizes["y"] == 337
    assert ndvi.t.values[0] == np.datetime64("2026-09-01")


def test_periods_stop_at_the_latest_published_dekad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clms_ndvi, "_search", lambda **_: [_item("2026-09-01", "s3://eodata/x/")])

    periods = asyncio.run(clms_ndvi.CLMSNDVIDekadalPlugin().periods("2026-08-01", "2026-12-31"))

    assert periods == ["2026-08-01", "2026-08-11", "2026-08-21", "2026-09-01"]


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (
            "s3://eodata/CLMS/ndvi/2026/09/01/c_gls_NDVI300_202609010000_GLOBE_OLCI_V3.0.1_cog/",
            "/vsis3/eodata/CLMS/ndvi/2026/09/01/c_gls_NDVI300_202609010000_GLOBE_OLCI_V3.0.1_cog/"
            "c_gls_NDVI300-NDVI_202609010000_GLOBE_OLCI_V3.0.1.tiff",
        ),
        (
            "s3://eodata/CLMS/ndvi/2015/06/01/c_gls_NDVI300_x_cog/c_gls_NDVI300-NDVI_201506010000_GLOBE_PROBAV_V3.0.1.tiff",
            "/vsis3/eodata/CLMS/ndvi/2015/06/01/c_gls_NDVI300_x_cog/c_gls_NDVI300-NDVI_201506010000_GLOBE_PROBAV_V3.0.1.tiff",
        ),
    ],
    ids=["folder", "file"],
)
def test_ndvi_path_handles_folder_and_file_assets(href: str, expected: str) -> None:
    assert clms_ndvi._ndvi_path(_item("2026-09-01", href)) == expected
