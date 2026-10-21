"""Population-weighted spatial aggregation of a climate cube to org units.

Replicates the DHIS2 Climate Tools "population weighting" guide
(https://climate-tools.dhis2.org/guides/aggregation/population-weighting/):

    weighted mean per org unit = sum(value * population) / sum(population)

`data` is the climate cube (t, y, x), `weights` is a population cube. The
aggregation runs on the population grid, which is normally much finer: a population
cell counts if its centre is inside the feature, and takes the climate value of the
climate cell it lies in. No population is resampled, so every person is counted
once. Population and features do not change between time steps, so each feature's
people per climate cell are summed once, and a step is then a weighted sum over a
few climate cells. Output matches `aggregate_spatial`: a dataset with (geometry, t)
dims that feeds the DHIS2 JSON / CHAP CSV exporters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from open_climate_service.process import process


def _parse_geometries(geometries: Any) -> tuple[list[Any], list[str]]:
    from shapely.geometry import shape

    if isinstance(geometries, dict):
        if geometries.get("type") == "FeatureCollection":
            feats = geometries.get("features", [])
            return [shape(f["geometry"]) for f in feats], [str(f.get("id", i)) for i, f in enumerate(feats)]
        if geometries.get("type") == "Feature":
            return [shape(geometries["geometry"])], [str(geometries.get("id", 0))]
        return [shape(geometries)], ["0"]
    items = [shape(g) if isinstance(g, dict) else g for g in geometries]
    return items, [str(i) for i in range(len(items))]


def _find_dim(obj: Any, candidates: list[str]) -> str | None:
    dims = obj.dims if isinstance(obj, xr.DataArray) else set(obj.dims)
    for c in candidates:
        if c in dims:
            return c
    return None


def _nearest_index(coords: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each target, the index of the nearest coordinate, or -1 outside the grid's outer cells."""
    order = np.argsort(coords)
    ordered = coords[order]
    if ordered.size == 1:
        return np.zeros(targets.size, dtype=np.intp)
    pos = np.clip(np.searchsorted(ordered, targets), 1, ordered.size - 1)
    pos = pos - ((targets - ordered[pos - 1]) <= (ordered[pos] - targets))
    half = abs(float(ordered[1] - ordered[0])) / 2
    outside = (targets < ordered[0] - half) | (targets > ordered[-1] + half)
    return np.where(outside, -1, order[pos])


def _cell_weights(
    geom: Any,
    people: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    cell_of_x: np.ndarray,
    cell_of_y: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The people of *geom* summed per climate cell: flat climate cell indices and their counts.

    *people* is north-up and west-to-east, with centres *px* and *py*. Only the window around the
    feature is rasterized, so a small unit does not cost a pass over a 100 m country grid.
    """
    import rasterio.features
    from rasterio.transform import from_origin
    from shapely.geometry import mapping

    dx = float(px[1] - px[0]) if px.size > 1 else 1.0
    dy = float(py[0] - py[1]) if py.size > 1 else 1.0
    minx, miny, maxx, maxy = geom.bounds
    c0 = max(0, int(np.floor((minx - (px[0] - dx / 2)) / dx)))
    c1 = min(px.size, int(np.ceil((maxx - (px[0] - dx / 2)) / dx)))
    r0 = max(0, int(np.floor(((py[0] + dy / 2) - maxy) / dy)))
    r1 = min(py.size, int(np.ceil(((py[0] + dy / 2) - miny) / dy)))
    if c1 <= c0 or r1 <= r0:
        return np.empty(0, dtype=np.intp), np.empty(0)
    transform = from_origin(px[c0] - dx / 2, py[r0] + dy / 2, dx, dy)
    inside = rasterio.features.geometry_mask(
        [mapping(geom)], out_shape=(r1 - r0, c1 - c0), transform=transform, invert=True
    )
    window = people[r0:r1, c0:c1]
    rows, cols = np.nonzero(inside & np.isfinite(window) & (window > 0))
    cy, cx = cell_of_y[r0 + rows], cell_of_x[c0 + cols]
    on_grid = (cy >= 0) & (cx >= 0)
    flat = cy[on_grid] * width + cx[on_grid]
    counts = np.bincount(flat, weights=window[rows[on_grid], cols[on_grid]])
    cells = np.nonzero(counts)[0]
    return cells, counts[cells]


def _as_dataarray(cube: Any) -> xr.DataArray:
    if isinstance(cube, xr.Dataset):
        return cube[list(cube.data_vars)[0]]
    if "bands" in getattr(cube, "dims", ()):
        return cube.isel(bands=0, drop=True)
    return cube


@process(
    summary="Population-weighted spatial aggregation to geometries",
    description=(
        "For each geometry computes sum(data * weights) / sum(weights) per time step — "
        "the population-weighted mean of the climate variable. It runs on the grid of `weights` "
        "(population): a population cell counts if its centre is inside the geometry, and takes "
        "the value of the climate cell it lies in. A time dimension on the weights is averaged "
        "to a single surface. Returns one value per geometry per "
        "climate time step."
    ),
)
def aggregate_spatial_population_weighted(
    data: Any,
    weights: Any,
    geometries: Any,
    target_dimension: str | None = None,
) -> xr.Dataset:
    """Return the population-weighted mean of `data` per geometry and time step, named after the climate variable."""
    data = _as_dataarray(data)
    weights = _as_dataarray(weights)

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    t_dim = _find_dim(data, ["t", "time"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial_population_weighted: cannot find x/y dims in {list(data.dims)}")

    # Reduce the population to a single surface, north-up and west-to-east, on its own grid.
    wx = _find_dim(weights, ["x", "longitude", "lon"])
    wy = _find_dim(weights, ["y", "latitude", "lat"])
    if wx is None or wy is None:
        raise ValueError(f"aggregate_spatial_population_weighted: cannot find x/y dims in {list(weights.dims)}")
    for d in list(weights.dims):
        if d not in (wx, wy):
            weights = weights.mean(dim=d) if weights.sizes[d] > 1 else weights.isel({d: 0}, drop=True)
    # A 100 m country grid is large, so it is reordered only when it has to be, and not widened.
    if weights[wx].size > 1 and float(weights[wx][0]) > float(weights[wx][-1]):
        weights = weights.isel({wx: slice(None, None, -1)})
    if weights[wy].size > 1 and float(weights[wy][0]) < float(weights[wy][-1]):
        weights = weights.isel({wy: slice(None, None, -1)})
    people = np.asarray(weights.transpose(wy, wx).values)
    px = weights[wx].values.astype(float)
    py = weights[wy].values.astype(float)

    # The climate cell each population row and column lies in, whatever the climate grid's order.
    x_coords = data[x_dim].values.astype(float)
    y_coords = data[y_dim].values.astype(float)
    width = len(x_coords)
    cell_of_x = _nearest_index(x_coords, px)
    cell_of_y = _nearest_index(y_coords, py)

    if t_dim:
        arr = data.transpose(t_dim, y_dim, x_dim).values.astype("float64")
    else:
        arr = data.transpose(y_dim, x_dim).values.astype("float64")[None]
    n_t = arr.shape[0]
    steps = arr.reshape(n_t, -1)

    geoms, labels = _parse_geometries(geometries)
    geom_dim = target_dimension or "geometry"

    out = np.full((len(geoms), n_t), np.nan, dtype="float64")
    for gi, geom in enumerate(geoms):
        cells, counts = _cell_weights(geom, people, px, py, cell_of_x, cell_of_y, width)
        if cells.size == 0:
            continue
        values = steps[:, cells]
        valid = np.isfinite(values)
        denom = np.where(valid, counts, 0.0).sum(axis=1)
        numer = np.where(valid, values * counts, 0.0).sum(axis=1)
        out[gi] = np.divide(numer, denom, out=np.full(n_t, np.nan), where=denom > 0)

    # Emit the period dimension as the canonical "t" (the CHAP/DHIS2 exporters
    # expect "t"), regardless of whether the source cube used "t" or "time".
    coords: dict[str, Any] = {geom_dim: labels}
    if t_dim:
        coords["t"] = data[t_dim].values
        result = xr.DataArray(out, dims=[geom_dim, "t"], coords=coords)
    else:
        result = xr.DataArray(out[:, 0], dims=[geom_dim], coords=coords)
    return result.to_dataset(name=data.name or "value")
