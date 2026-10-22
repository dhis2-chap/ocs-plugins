"""remap_classes merges a categorical raster's classes into groups that aggregate_spatial_classes reads.

Grids are small and synthetic, unit cells from (0, 0) in WGS 84, so every share can be worked out
by hand. The aggregation is OCS core's, unchanged: the groups reach it as the cube's CF flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.openeo import plugin_processes, workflows
from open_climate_service.plugins.processes.aggregate_spatial_classes import aggregate_spatial_classes
from open_climate_service.shared.classes import classification_from_template, cube_classes

from ocs_plugins.processes.remap_classes import remap_classes

WORKFLOW = Path(__file__).resolve().parents[1] / "ocs_plugins" / "workflows" / "esacci_landcover_groups.json"
# Three forest classes and one cropland class, as ESA CCI codes, with their CF flags.
CLASSES = {"flag_values": [10, 50, 60, 70], "flag_meanings": "Cropland Tree_a Tree_b Tree_c", "long_name": "LCCS"}
GROUPS = {"cropland": [10], "tree cover": [50, 60, 70]}


def _grid(values: list[list[float]]) -> xr.DataArray:
    """Unit cells from (0, 0), rows listed north to south, carrying the ESA CCI-like classes."""
    rows, cols = len(values), len(values[0])
    raster = xr.DataArray(
        np.array(values, dtype="float32"),
        dims=("y", "x"),
        coords={"y": (np.arange(rows) + 0.5)[::-1], "x": np.arange(cols) + 0.5},
        name="lccs_class",
        attrs=dict(CLASSES),
    )
    return raster.rio.write_crs("EPSG:4326")


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, Any]:
    ring = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
    return {"type": "Polygon", "coordinates": [ring]}


def test_codes_become_groups_and_the_groups_become_the_classes() -> None:
    out = remap_classes(_grid([[10, 50, 60, 70], [90, float("nan"), 10, 50]]), GROUPS)

    np.testing.assert_equal(out.values, [[1, 2, 2, 2], [np.nan, np.nan, 1, 2]])  # 90 is in no group
    assert cube_classes(out) == ([1, 2], ["cropland", "tree_cover"])
    assert "long_name" not in out.attrs  # it described the original classes
    assert out.rio.crs.to_epsg() == 4326 and out.name == "lccs_class"


def test_a_group_share_is_the_sum_of_its_classes_shares() -> None:
    raster = _grid([[10, 10, 50, 50], [10, 10, 60, 60], [10, 10, 70, 70], [10, 10, 10, 10]])
    whole = _box(0, 0, 4, 4)

    by_class = aggregate_spatial_classes(raster, whole, "share").isel(geometry=0)
    by_group = aggregate_spatial_classes(remap_classes(raster, GROUPS), whole, "share").isel(geometry=0)

    assert by_group["class_label"].values.tolist() == ["cropland", "tree_cover"]
    shares = dict(zip(by_group["class_label"].values.tolist(), by_group.values.tolist(), strict=True))
    assert shares["cropland"] == pytest.approx(10 / 16)
    assert shares["tree_cover"] == pytest.approx(float(by_class.sel({"class": [50, 60, 70]}).sum()))


def test_the_majority_is_the_group_covering_most() -> None:
    # Cropland is the largest single class (6 of 16 cells), but the three tree classes cover 10.
    raster = _grid([[10, 10, 50, 50], [10, 10, 60, 60], [10, 10, 70, 70], [50, 60, 70, 50]])

    assert int(aggregate_spatial_classes(raster, _box(0, 0, 4, 4), "majority").values.item()) == 10
    grouped = aggregate_spatial_classes(remap_classes(raster, GROUPS), _box(0, 0, 4, 4), "majority")
    assert int(grouped.values.item()) == 2  # tree_cover


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({}, "groups is empty"),
        ({"a": [10], "b": [10]}, "class 10 is in more than one group"),
        ({"a": [10], "A ": [50], "a!": [60]}, "group names must differ"),
        ({"!!!": [10]}, "has no letters or digits"),
    ],
)
def test_malformed_groups_are_refused(groups: dict[str, list[int]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        remap_classes(_grid([[10]]), groups)


def test_the_process_and_the_workflow_are_discovered_through_the_entry_point() -> None:
    assert "remap_classes" in {func.__name__ for func in plugin_processes._scan_plugin_package_processes()}
    assert "esacci_landcover_groups" in {record["id"] for record in workflows._load_entry_point_workflows()}


def test_the_esa_cci_groups_cover_every_class_of_the_core_dataset_once() -> None:
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))["process_graph"]
    groups = graph["groups"]["arguments"]["groups"]
    template = registry.get_dataset("esacci_landcover_class_yearly")
    classification = classification_from_template(template)
    assert classification is not None

    grouped = [code for codes in groups.values() for code in codes]
    assert sorted(grouped) == sorted(classification.values)
    assert len(groups) == 10
