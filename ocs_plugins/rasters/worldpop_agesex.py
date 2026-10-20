"""WorldPop Global2 population by age and sex, from the plugin in OCS core.

OCS keeps the plugin but no longer lists the dataset, since its ingest is slow: WorldPop publishes
about 40 GeoTIFFs per country-year, one per sex and age band, and each is downloaded and stacked.
The template here offers it to the instances that want it anyway.
"""

from open_climate_service.plugins.rasters.worldpop import WorldPopAgeSexYearlyPlugin

__all__ = ["WorldPopAgeSexYearlyPlugin"]
