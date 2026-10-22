"""remap_classes: merge the classes of a categorical raster into groups.

A categorical cube carries its classes as CF flags (``flag_values`` and ``flag_meanings``), which
``aggregate_spatial_classes`` reads to know the codes and their labels. Remapping the codes with
standard processes (``apply`` with ``array_element``) keeps the original flags, so the aggregation
would look for codes the raster no longer holds. This process remaps the codes and replaces the
flags with the groups, so ``aggregate_spatial_classes`` reports one share, total or majority per
group, labelled with the group names.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import xarray as xr
from open_climate_service.process import process

_DESCRIBING_ATTRS = ("flag_values", "flag_meanings", "flag_masks", "standard_name", "long_name", "description")
"""Attributes that describe the original classes, and are wrong once the codes are groups."""


def _group_name(label: str) -> str:
    """A group label as one CF flag meaning: no spaces, as ``cf_flag_attrs`` writes them."""
    name = re.sub(r"[^0-9A-Za-z_.+@-]+", "_", label).strip("_")
    if not name:
        raise ValueError(f"remap_classes: group label {label!r} has no letters or digits")
    return name


def _class_code(value: Any) -> int:
    """A class code as an int, refusing a value that only converts to one: 10.9 or ``true``."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
        raise ValueError(f"remap_classes: class code {value!r} is not a whole number")
    if value < 0:
        raise ValueError(f"remap_classes: class code {value} is negative")
    return int(value)


def _lookup(groups: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The sorted class codes, each one's group code (1, 2, ... in order), and the group names.

    Sorted codes keep the table as small as the mapping, whatever the codes' values.
    """
    if not groups:
        raise ValueError("remap_classes: groups is empty")
    group_of: dict[int, int] = {}
    names: list[str] = []
    for group_code, (label, codes) in enumerate(groups.items(), start=1):
        names.append(_group_name(str(label)))
        for value in codes:
            code = _class_code(value)
            if code in group_of:
                raise ValueError(f"remap_classes: class {code} is in more than one group")
            group_of[code] = group_code
    if len(set(names)) != len(names):
        raise ValueError(f"remap_classes: group names must differ, got {names}")
    codes = np.array(sorted(group_of), dtype="float64")
    group_codes = np.array([group_of[int(code)] for code in codes], dtype="float32")
    return codes, group_codes, names


@process(
    summary="Merge the classes of a categorical raster into groups",
    description=(
        "Replaces each class code with the code of its group, numbered 1, 2, ... in the order the "
        "groups are given, and records the groups as the cube's classes, so aggregate_spatial_classes "
        "reports one value per group, labelled with the group name. A class in no group, and a "
        "missing cell, become missing. `groups` maps each group name to its class codes."
    ),
)
def remap_classes(data: Any, groups: dict[str, list[int]]) -> xr.DataArray:
    """Return *data* with its class codes replaced by group codes, and the groups as its CF flags."""
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError(f"remap_classes: expected a single-variable raster, got {list(data.data_vars)}")
        data = data[next(iter(data.data_vars))]
    codes, group_codes, names = _lookup(groups)

    def remap(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype="float32")
        finite = np.isfinite(values)
        position = np.searchsorted(codes, values[finite])
        found = position < codes.size
        found[found] = codes[position[found]] == values[finite][found]
        mapped = np.full(position.shape, np.nan, dtype="float32")
        mapped[found] = group_codes[position[found]]
        out[finite] = mapped
        return out

    result: xr.DataArray = xr.apply_ufunc(remap, data, dask="parallelized", output_dtypes=[np.float32])
    attrs = {key: value for key, value in data.attrs.items() if key not in _DESCRIBING_ATTRS}
    attrs.update(units="1", flag_values=list(range(1, len(names) + 1)), flag_meanings=" ".join(names))
    result.attrs = attrs
    return result.rename(data.name)
