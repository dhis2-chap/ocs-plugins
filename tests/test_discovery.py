"""Every template here reaches an OCS instance through the entry point, and can run.

Through OCS's own registry rather than by reading the YAML, so this fails the way an instance
would: a template OCS refuses, a plugin path that does not import, or a licence it cannot read.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from open_climate_service.data_registry.services import datasets as registry
from open_climate_service.shared.licences import UNDECLARED, parse_licence
from open_climate_service.streaming import BaseDatasetPlugin

RASTERS = Path(__file__).resolve().parents[1] / "ocs_plugins" / "rasters"
TEMPLATE_IDS = sorted(
    entry["id"] for path in RASTERS.glob("*.yaml") for entry in yaml.safe_load(path.read_text(encoding="utf-8"))
)


@pytest.fixture(scope="module")
def discovered() -> dict[str, dict[str, Any]]:
    registry.reset_template_caches()
    return {dataset["id"]: dataset for dataset in registry.list_datasets()}


def test_every_template_is_discovered_through_the_entry_point(discovered: dict[str, dict[str, Any]]) -> None:
    assert set(TEMPLATE_IDS) <= set(discovered)


@pytest.mark.parametrize("dataset_id", TEMPLATE_IDS)
def test_each_plugin_class_imports_and_is_a_dataset_plugin(
    discovered: dict[str, dict[str, Any]], dataset_id: str
) -> None:
    path = discovered[dataset_id]["ingestion"]["plugin"]
    assert path.startswith("ocs_plugins.rasters."), path
    module, _, name = path.rpartition(".")
    plugin = getattr(importlib.import_module(module), name)
    assert issubclass(plugin, BaseDatasetPlugin)


@pytest.mark.parametrize("dataset_id", TEMPLATE_IDS)
def test_each_template_declares_a_readable_licence(discovered: dict[str, dict[str, Any]], dataset_id: str) -> None:
    assert parse_licence(discovered[dataset_id].get("license")) is not UNDECLARED
