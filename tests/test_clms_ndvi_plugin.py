"""CLMS NDVI and the shared CDSE reader: asset paths, window reads, scaling and flags.

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

from ocs_plugins.rasters import _clms, clms_ndvi

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


# NDVI


@pytest.fixture
def ndvi_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = np.full((SIZE, SIZE), 200, dtype="uint8")  # 0.004 * 200 - 0.08 = 0.72
    raw[:, :10] = 254  # water along the western edge
    raw[0, :] = 255  # missing along the top row
    path = _write(tmp_path / "ndvi.tiff", raw, scale=0.004, offset=-0.08, nodata=255)
    monkeypatch.setattr(_clms, "search", Catalogue([_item("c_gls_NDVI300_x_cog", "2026-09-01", {})]))
    monkeypatch.setattr(_clms, "asset_path", lambda item, variable: str(path))
    return path


def test_ndvi_is_scaled_with_the_files_own_scale_and_flags_are_empty(ndvi_file: Path) -> None:
    ds = clms_ndvi.CLMSNDVIDekadalPlugin().fetch_period("2026-09-01", [28.0, -3.0, 29.0, 0.0])
    ndvi = ds["ndvi"].isel(t=0)

    assert ds.rio.crs.to_epsg() == 4326
    assert np.isnan(ndvi.isel(x=slice(0, 10))).all()
    assert np.isnan(ndvi.isel(y=0)).all()
    assert np.allclose(ndvi.isel(y=slice(1, None), x=slice(10, None)), 0.72)


def test_the_window_is_on_the_product_grid(ndvi_file: Path) -> None:
    ndvi = clms_ndvi.CLMSNDVIDekadalPlugin().fetch_period("2026-09-01", BBOX)["ndvi"]

    assert np.allclose(ndvi.x.values * 336, np.round(ndvi.x.values * 336))
    assert np.allclose(ndvi.y.values * 336, np.round(ndvi.y.values * 336))
    assert ndvi.sizes["x"] == ndvi.sizes["y"] == 337
    assert float(ndvi.x.min()) <= BBOX[0] + HALF and float(ndvi.x.max()) >= BBOX[2] - HALF
    assert ndvi.t.values[0] == np.datetime64("2026-09-01")


def test_ndvi_periods_stop_at_the_latest_dekad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_clms, "search", Catalogue([_item("c_gls_NDVI300_x_cog", "2026-09-01", {})]))

    periods = asyncio.run(clms_ndvi.CLMSNDVIDekadalPlugin().periods("2026-08-01", "2026-12-31"))

    assert periods == ["2026-08-01", "2026-08-11", "2026-08-21", "2026-09-01"]


# Asset paths


@pytest.mark.parametrize(
    ("variable", "asset_key", "href", "expected"),
    [
        (
            "ndvi",
            "ndvi",
            "s3://eodata/CLMS/ndvi/2026/09/01/c_gls_NDVI300_202609010000_GLOBE_OLCI_V3.0.1_cog/",
            "/vsis3/eodata/CLMS/ndvi/2026/09/01/c_gls_NDVI300_202609010000_GLOBE_OLCI_V3.0.1_cog/"
            "c_gls_NDVI300-NDVI_202609010000_GLOBE_OLCI_V3.0.1.tiff",
        ),
        (
            "ndvi",
            "ndvi",
            "s3://eodata/CLMS/ndvi/2015/06/01/x_cog/c_gls_NDVI300-NDVI_201506010000_GLOBE_PROBAV_V3.0.1.tiff",
            "/vsis3/eodata/CLMS/ndvi/2015/06/01/x_cog/c_gls_NDVI300-NDVI_201506010000_GLOBE_PROBAV_V3.0.1.tiff",
        ),
        (
            "npp",
            "npp300_npp",
            "s3://eodata/CLMS/npp/2025/06/01/x_cog/c_gls_NPP300-NPP-RT6_202506100000_GLOBE_OLCI_V2.0.2.tiff",
            "/vsis3/eodata/CLMS/npp/2025/06/01/x_cog/c_gls_NPP300-NPP-RT6_202506100000_GLOBE_OLCI_V2.0.2.tiff",
        ),
    ],
    ids=["folder", "file", "product-prefixed-key"],
)
def test_asset_path_handles_each_layout(variable: str, asset_key: str, href: str, expected: str) -> None:
    alternate = (
        "https://download.example/odata/v1/Products(x)/Nodes(x_cog)/"
        "Nodes(c_gls_NDVI300-NDVI_202609010000_GLOBE_OLCI_V3.0.1.tiff)/$value"
    )
    item = _item("x", "2026-09-01", {asset_key: {"href": href, "alternate": {"https": {"href": alternate}}}})

    assert _clms.asset_path(item, variable) == expected


def test_a_missing_asset_is_named() -> None:
    with pytest.raises(RuntimeError, match="no 'gpp' asset"):
        _clms.asset_path(_item("x", "2026-09-01", {"qflag": {"href": "s3://x"}}), "gpp")


# S3 credentials


_S3_READ_OPTIONS = {"AWS_VIRTUAL_HOSTING": "FALSE", "AWS_HTTPS": "YES"}


def test_environment_keys_reach_gdal_through_a_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "access")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "secret")

    session = _clms._s3_session()
    with rasterio.Env(session, **_S3_READ_OPTIONS):  # rasterio refuses keys passed as plain options
        pass

    options = session.get_credential_options()
    assert options["AWS_ACCESS_KEY_ID"] == "access" and options["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert options["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu"


def test_the_cdse_profile_is_used_without_environment_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = tmp_path / "credentials"
    credentials.write_text("[cdse]\naws_access_key_id = profile-access\naws_secret_access_key = profile-secret\n")
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))

    options = _clms._s3_session().get_credential_options()

    assert options["AWS_ACCESS_KEY_ID"] == "profile-access"


def test_missing_keys_say_how_to_get_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))

    with pytest.raises(PermissionError, match="eodata-s3keysmanager"):
        _clms._s3_session()
