"""MODIS NRT flood: the days offered are the days asked for."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ocs_plugins.rasters import mcdwd


def test_a_composite_starting_the_day_before_is_not_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search finds the 2-day composite that starts the day before, as it overlaps the day asked for."""
    today = datetime.now(UTC).date()
    day, before = today - timedelta(days=3), today - timedelta(days=4)

    def granules(self: Any, first: Any, last: Any, bbox: list[float]) -> list[dict[str, str]]:
        return [{"time_start": f"{before}T00:00:00Z"}, {"time_start": f"{day}T00:00:00Z"}]

    monkeypatch.setattr(mcdwd.MCDWDFloodDailyPlugin, "_granules", granules)
    monkeypatch.setattr("open_climate_service.extents.services.get_extent", lambda: None)

    assert asyncio.run(mcdwd.MCDWDFloodDailyPlugin().periods(str(day), str(day))) == [str(day)]
