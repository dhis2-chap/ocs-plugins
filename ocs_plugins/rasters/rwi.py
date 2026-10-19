"""Meta Relative Wealth Index, ingested on its native Web Mercator grid.

A socioeconomic covariate to sit alongside the climate data: wealth is a standard confounder
in health modelling and there was nothing of the kind in the catalogue.

VALUES ARE RELATIVE WITHIN ONE COUNTRY

RWI ranks micro-regions against others *in the same country*, not globally. So two OCS
instances cannot compare RWI values, a multi-country mosaic is meaningless, and an anomaly or
trend against another country's RWI is nonsense. This is a unit-like trap: it produces
plausible-looking output when got wrong, which is why it is repeated in the template
description (and so reaches the STAC collection) rather than living only here.

IT IS A RASTER, DESPITE ARRIVING AS POINTS

The CSV is a point list, which makes it look like vector data. It is not — it is a regular
grid that has had its empty cells omitted. Verified against the Nepal file:

    zoom-14 Bing/Web Mercator tiles, cell 2*pi*R/2^14 = 2445.985 m
    371 unique x, 196 unique y; every gap an integer multiple of the cell size
    full grid 200 x 371 = 74,200 cells, of which 18,291 (24.7 %) are populated

Crucially the grid is regular **in EPSG:3857 only**. Latitude gaps in degrees are not integer
multiples of anything, because Mercator y is not linear in latitude — checked, and they are
not. So the grid is reconstructed in Mercator and the store keeps EPSG:3857 as its CRS
(`crs = 3857`; `BaseDatasetPlugin` supports a projected source, and the ingest clips in the
source CRS rather than reprojecting). The spatial aggregations reproject org-unit geometries
onto this grid, so it aggregates like any other store.

RESOLVING THE COUNTRY FILE

HDX filenames are *mostly* lowercase ISO3, but not reliably: of 109 country files, 92 are a
plain ISO3 stem, one is `ind_pak` (India and Pakistan share a file, so ISO3 to file is not
1:1), and 16 are country-name slugs like `bosnia-and-herzegovina`. Resolution therefore tries
ISO3 stems and ISO3 tokens, and falls back to an explicit `resource_filename` override rather
than guessing from the resource title.

Not every country is covered — the product is low- and middle-income countries only, so
Norway is absent. A missing country fails with a message naming it and listing near matches,
rather than a bare 404 from the download.

LICENCE AND ATTRIBUTION

HDX lists the licence as "Other" with no licence URL, so this is **not** a clean CC BY;
confirm the terms before publishing. Attribution is required: Chi, Fang, Chatterjee &
Blumenstock, PNAS 2022, doi:10.1073/pnas.2113658119.

The citation is stored as CF `references`, which OCS publishes to `cube:variables`
(CLIM-973). The licence is declared on the template and published as the STAC collection
licence (CLIM-946) — CF has no licence attribute, so it does not belong on the variable.
"""

from __future__ import annotations

import io
import logging
import math
import re
from typing import Any

import httpx
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # registers the .rio accessor
import xarray as xr
from open_climate_service.streaming import BaseDatasetPlugin

# Under the `open_climate_service` tree: startup.py attaches the service's only INFO handler
# there and sets propagate=False, so a plugin logging under its own module name is silent in
# the service log.
logger = logging.getLogger(f"open_climate_service.plugins.{__name__}")

_HDX_PACKAGE = "https://data.humdata.org/api/3/action/package_show"
_HDX_DATASET_ID = "relative-wealth-index"
_SUFFIX = "_relative_wealth_index.csv"

# Bing/Web Mercator zoom 14. The cell is square in EPSG:3857 by construction.
_ZOOM = 14
_EARTH_CIRCUMFERENCE_M = 2.0 * math.pi * 6378137.0
_CELL_M = _EARTH_CIRCUMFERENCE_M / (2**_ZOOM)

# Snapping tolerance when placing a point on the reconstructed grid, as a fraction of a cell.
# The published coordinates are cell centres rounded to six decimals, so they land a few
# metres off an exactly-computed grid; a quarter cell is far more than that slop and far less
# than a neighbouring cell.
_SNAP_TOLERANCE = 0.25

# The two measures, as the `measure` coordinate. Named rather than positional so a consumer
# reading the cube does not have to know which index is which.
_MEASURES = ("index", "error")

# The one release (April 2021), as the yearly period OCS expects.
_YEAR = "2021"


def _hdx_resources() -> list[dict[str, Any]]:
    response = httpx.get(_HDX_PACKAGE, params={"id": _HDX_DATASET_ID}, timeout=120, follow_redirects=True)
    response.raise_for_status()
    resources = response.json()["result"]["resources"]
    return [r for r in resources if isinstance(r, dict)]


def _filename(url: str | None) -> str:
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def _resolve_resource(country_code: str, override: str | None = None) -> tuple[str, str]:
    """Find the HDX download URL for a country, or explain why there isn't one.

    Returns ``(url, filename)``. Matching is on the *filename*, not the resource title: titles
    are English country names ("Nepal_relative_wealth_index.csv") while filenames carry the
    ISO3 stem, and only the latter can be derived from `extent.country_code`.
    """
    resources = _hdx_resources()
    by_name = {_filename(r.get("url")): r for r in resources}

    if override:
        if override not in by_name:
            raise ValueError(
                f"resource_filename {override!r} is not in the RWI release. "
                f"Available country files: {sorted(n for n in by_name if n.endswith(_SUFFIX))[:8]}..."
            )
        return str(by_name[override]["url"]), override

    iso3 = country_code.strip().lower()
    if not re.fullmatch(r"[a-z]{3}", iso3):
        raise ValueError(f"country_code must be an ISO 3166-1 alpha-3 code, got {country_code!r}")

    country_files = [n for n in by_name if n.endswith(_SUFFIX)]
    exact = f"{iso3}{_SUFFIX}"
    if exact in by_name:
        return str(by_name[exact]["url"]), exact

    # India and Pakistan share `ind_pak_relative_wealth_index.csv`, so a country may appear as
    # one token of a multi-country stem.
    for name in country_files:
        stem = name[: -len(_SUFFIX)]
        if re.fullmatch(r"[a-z]{3}(_[a-z]{3})+", stem) and iso3 in stem.split("_"):
            logger.info("%s resolves to the shared file %s", iso3.upper(), name)
            return str(by_name[name]["url"]), name

    # 16 of 109 files use a country-name slug rather than ISO3, so failure here is not
    # necessarily "country absent" — it may be "country present under a name we cannot derive".
    # Say both, and name the escape hatch.
    slugs = sorted(n for n in country_files if not re.fullmatch(r"[a-z]{3}(_[a-z]{3})*", n[: -len(_SUFFIX)]))
    raise ValueError(
        f"No Relative Wealth Index file for country_code {country_code.upper()!r}. The product "
        f"covers low- and middle-income countries only (~93), so high-income countries such as "
        f"Norway are genuinely absent. Note also that {len(slugs)} files are named by country "
        f"slug rather than ISO3 and cannot be resolved from a country code — if this country is "
        f"one of them, set `resource_filename` in the template's ingestion params. Slug-named "
        f"files: {[s[: -len(_SUFFIX)] for s in slugs]}"
    )


def _to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat degrees to EPSG:3857 metres.

    Done with the closed-form spherical formulae rather than pyproj so the grid maths matches
    the Bing tiling definition exactly; Web Mercator is spherical by definition, and a
    datum-aware transform would introduce a sub-metre disagreement with the tile grid.
    """
    x = np.radians(lon) * 6378137.0
    y = np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0)) * 6378137.0
    return x, y


class RelativeWealthIndexPlugin(BaseDatasetPlugin):
    """Meta RWI for one country, on the native zoom-14 Web Mercator grid.

    Params:

    - `country_code` — ISO3; defaults to the instance's `extent.country_code`
    - `resource_filename` — override for the 16 slug-named files
    - `variable` — stored variable name (default `rwi`)
    """

    max_concurrency = 1
    commit_batch_size = 1
    crs = 3857

    def __init__(
        self,
        country_code: str | None = None,
        resource_filename: str | None = None,
        variable: str = "rwi",
        **_: Any,
    ) -> None:
        self.country_code = country_code
        self.resource_filename = resource_filename
        self.variable = str(variable)

    async def periods(self, start: str, end: str) -> list[str]:
        return [_YEAR] if int(start[:4]) <= int(_YEAR) <= int(end[:4]) else []

    def _country(self, params: dict[str, Any]) -> str:
        code = self.country_code or params.get("country_code")
        if not code:
            raise ValueError(
                "RWI needs a country: set `country_code` in the template's ingestion params, or "
                "`extent.country_code` on the instance. The product is published per country and "
                "its values are only comparable within one."
            )
        return str(code)

    def fetch_period(self, period_id: str, bbox: list[float], **params: Any) -> xr.Dataset:
        country = self._country(params)
        url, filename = _resolve_resource(country, self.resource_filename)
        logger.info("RWI: %s -> %s", country.upper(), filename)

        response = httpx.get(url, timeout=300, follow_redirects=True)
        response.raise_for_status()
        frame = pd.read_csv(io.BytesIO(response.content))
        missing = {"latitude", "longitude", "rwi", "error"} - set(frame.columns)
        if missing:
            raise ValueError(f"{filename}: missing expected column(s) {sorted(missing)}")

        west, south, east, north = (float(v) for v in bbox)
        # Clip in lon/lat before projecting: the file is a whole country and the instance
        # extent may be smaller, and this is the only place the two are expressed alike.
        inside = frame.longitude.between(west, east) & frame.latitude.between(south, north)
        frame = frame.loc[inside]
        if frame.empty:
            raise ValueError(
                f"No RWI points from {filename} fall inside the configured extent {[west, south, east, north]}"
            )

        mx, my = _to_mercator(frame.longitude.to_numpy(), frame.latitude.to_numpy())

        # Reconstruct the full regular grid. The file omits unpopulated cells, so the sparse
        # points are placed onto a complete grid and everything else left NaN — never 0, which
        # would read as "poorest" rather than "unknown".
        x0, y0 = mx.min(), my.min()
        ix = np.round((mx - x0) / _CELL_M)
        iy = np.round((my - y0) / _CELL_M)
        drift = max(
            float(np.abs((mx - x0) / _CELL_M - ix).max()),
            float(np.abs((my - y0) / _CELL_M - iy).max()),
        )
        if drift > _SNAP_TOLERANCE:
            # If the points do not sit on a zoom-14 lattice then this is not the grid the
            # module assumes, and silently binning them would fabricate a plausible raster.
            raise ValueError(
                f"{filename}: points do not lie on a zoom-{_ZOOM} Mercator lattice "
                f"(worst offset {drift:.3f} of a {_CELL_M:.1f} m cell). The release layout may "
                f"have changed; reconstructing a grid from it would be guesswork."
            )
        ix = ix.astype(int)
        iy = iy.astype(int)

        width = int(ix.max()) + 1
        height = int(iy.max()) + 1
        cube = np.full((len(_MEASURES), height, width), np.nan, dtype="float32")
        cube[0, iy, ix] = frame.rwi.to_numpy(dtype="float32")
        cube[1, iy, ix] = frame.error.to_numpy(dtype="float32")

        xs = x0 + np.arange(width) * _CELL_M
        ys = y0 + np.arange(height) * _CELL_M
        # y descending, the orientation the rest of OCS and the viewer expect.
        cube = cube[:, ::-1, :]
        ys = ys[::-1]

        populated = 100.0 * float(np.isfinite(cube[0]).mean())
        logger.info(
            "%s: %d points on a %d x %d zoom-%d grid (%.1f%% populated), cell %.1f m, EPSG:3857",
            filename,
            len(frame),
            height,
            width,
            _ZOOM,
            populated,
            _CELL_M,
        )

        data = xr.DataArray(
            cube,
            coords={"measure": list(_MEASURES), "y": ys, "x": xs},
            dims=("measure", "y", "x"),
        ).rio.write_crs(3857)

        cubeset = data.expand_dims(t=[np.datetime64(f"{_YEAR}-01-01", "ns")]).to_dataset(name=self.variable)
        cubeset[self.variable].attrs.update(
            {
                "long_name": "Meta Relative Wealth Index",
                # Genuinely unitless: an index, not a measurement in any unit.
                "units": "1",
                # CF section 2.6.2's `references` — "published or web-based references that
                # describe the data" — rather than an invented `attribution`. CF names it, CF
                # allows it on a variable, and OCS passes exactly those four through to
                # `cube:variables` (CLIM-973), so a standard name is the difference between
                # this reaching a client and being stripped.
                "references": ("Chi, Fang, Chatterjee & Blumenstock, PNAS 2022, doi:10.1073/pnas.2113658119"),
                "comment": (
                    "Relative WITHIN one country: values rank micro-regions against others in "
                    f"the same country ({country.upper()}), not globally. Not comparable across "
                    "countries or instances. `measure='index'` is the wealth index, "
                    "`measure='error'` its prediction uncertainty. Unpopulated cells are NaN, "
                    "not 0."
                ),
            }
        )
        return cubeset
