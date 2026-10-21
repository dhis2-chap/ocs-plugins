"""The population-weighted aggregation process reaches OCS through the entry point and weights by population."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr
from open_climate_service.openeo import plugin_processes

from ocs_plugins.processes.aggregate_spatial_population_weighted import aggregate_spatial_population_weighted

# A 2x2 grid of 1-degree cells, its centres at x 0.5 and 1.5, y 1.5 and 0.5 (north up).
X = [0.5, 1.5]
Y = [1.5, 0.5]
WHOLE_GRID = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
LEFT_COLUMN = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 2], [0, 2], [0, 0]]]}


def _units(*geometries: dict[str, object]) -> dict[str, object]:
    features = [{"type": "Feature", "id": f"unit{i}", "geometry": g} for i, g in enumerate(geometries)]
    return {"type": "FeatureCollection", "features": features}


def _grid(values: list[list[float]], times: list[str] | None = None) -> xr.DataArray:
    if times is None:
        return xr.DataArray(np.array(values, dtype="float64"), dims=("y", "x"), coords={"y": Y, "x": X})
    data = np.array([values] * len(times), dtype="float64")
    t = np.array(times, dtype="datetime64[ns]")
    return xr.DataArray(data, dims=("t", "y", "x"), coords={"t": t, "y": Y, "x": X})


def test_the_process_is_discovered_through_the_entry_point() -> None:
    processes = {func.__name__ for func in plugin_processes._scan_plugin_package_processes()}
    assert "aggregate_spatial_population_weighted" in processes


def test_the_mean_is_weighted_by_population() -> None:
    climate = _grid([[30.0, 20.0], [20.0, 20.0]], times=["2017-01-01", "2017-01-02"])
    population = _grid([[300.0, 100.0], [0.0, 0.0]])

    result = aggregate_spatial_population_weighted(climate, population, _units(WHOLE_GRID))["value"]

    # (30 * 300 + 20 * 100) / 400, where the plain area mean would be 22.5.
    assert result.sel(geometry="unit0").values.tolist() == [27.5, 27.5]
    assert list(result.dims) == ["geometry", "t"]


def test_cells_without_data_or_people_are_left_out_and_an_empty_unit_is_missing() -> None:
    climate = _grid([[np.nan, 20.0], [25.0, 20.0]], times=["2017-01-01"])
    population = _grid([[500.0, 100.0], [100.0, 0.0]])

    result = aggregate_spatial_population_weighted(climate, population, _units(WHOLE_GRID, LEFT_COLUMN))["value"]

    # The cell with no data is skipped rather than counted as 0: (20 * 100 + 25 * 100) / 200.
    assert result.sel(geometry="unit0").values.tolist() == [22.5]
    assert result.sel(geometry="unit1").values.tolist() == [25.0]

    nobody = _grid([[0.0, 0.0], [0.0, 0.0]])
    empty = aggregate_spatial_population_weighted(climate, nobody, _units(LEFT_COLUMN))["value"]
    assert math.isnan(float(empty.sel(geometry="unit0").values[0]))


def _fine(people: dict[tuple[float, float], float]) -> xr.DataArray:
    """A 4x4 population grid of half-degree cells over the 2x2 climate grid, north up."""
    xs, ys = [0.25, 0.75, 1.25, 1.75], [1.75, 1.25, 0.75, 0.25]
    values = np.zeros((4, 4))
    for (x, y), count in people.items():
        values[ys.index(y), xs.index(x)] = count
    return xr.DataArray(values, dims=("y", "x"), coords={"y": ys, "x": xs})


def test_every_person_on_a_finer_population_grid_counts() -> None:
    climate = _grid([[30.0, 20.0], [20.0, 20.0]], times=["2017-01-01"])
    # Off the climate cell centres, where sampling the population at those centres finds nobody.
    population = _fine({(0.25, 1.75): 300.0, (1.25, 1.75): 100.0})

    result = aggregate_spatial_population_weighted(climate, population, _units(WHOLE_GRID))["value"]

    assert result.sel(geometry="unit0").values.tolist() == [27.5]


def test_a_feature_smaller_than_a_climate_cell_takes_its_people() -> None:
    climate = _grid([[30.0, 20.0], [20.0, 20.0]], times=["2017-01-01"])
    population = _fine({(0.25, 1.75): 50.0})
    # One population cell, holding no climate cell centre.
    corner = {"type": "Polygon", "coordinates": [[[0, 1.5], [0.5, 1.5], [0.5, 2], [0, 2], [0, 1.5]]]}

    result = aggregate_spatial_population_weighted(climate, population, _units(corner))["value"]

    assert result.sel(geometry="unit0").values.tolist() == [30.0]


def test_population_stored_x_first_on_a_rectangular_grid() -> None:
    xs, ys = [0.5, 1.5, 2.5], [1.5, 0.5]
    climate = xr.DataArray(
        np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]]),
        dims=("t", "y", "x"),
        coords={"t": np.array(["2017-01-01"], dtype="datetime64[ns]"), "y": ys, "x": xs},
    )
    people = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 0.0]])
    population = xr.DataArray(people.T, dims=("x", "y"), coords={"x": xs, "y": ys})
    everything = {"type": "Polygon", "coordinates": [[[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]]]}

    result = aggregate_spatial_population_weighted(climate, population, _units(everything))["value"]

    # All the people are in the north-east cell.
    assert result.sel(geometry="unit0").values.tolist() == [30.0]


def test_grids_running_east_to_west_keep_west_in_the_west() -> None:
    climate = _grid([[30.0, 20.0], [20.0, 20.0]], times=["2017-01-01"]).isel(x=slice(None, None, -1))
    population = _grid([[100.0, 100.0], [100.0, 100.0]]).isel(x=slice(None, None, -1))

    result = aggregate_spatial_population_weighted(climate, population, _units(LEFT_COLUMN))["value"]

    # The western column is 30 above 20.
    assert result.sel(geometry="unit0").values.tolist() == [25.0]
