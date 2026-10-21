# ocs-plugins

Experimental dataset plugins and processes for the [Open Climate Service](https://github.com/dhis2/open-climate-service)
(OCS), for testing.

> **Not for production.** Try these plugins and tell us what you find, but do not rely on them in
> a production instance. They follow OCS `main` and can change or be removed without notice: a
> dataset's id, variable or grid may change between versions, so a store may need ingesting again.

OCS core holds datasets every instance can rely on: actively maintained and updated upstream
sources from open providers, used beyond one country. This repository is where other datasets are
tried out: ones that are more specialised, read through commercial platforms, or near the end of
their record. They install into any OCS instance as one package, and the ones that prove
themselves can move to core.

## Try the plugins in an instance

The plugins are an installable OCS plugin package: once installed, OCS finds them through the
`open_climate_service.plugins` entry point and lists their datasets under **Data sources**,
next to the built-in ones. No `plugins_dir` change is needed.

Run OCS from its `main` branch, as the plugins follow it; the release on PyPI is behind. In the
instance's `pyproject.toml`:

```toml
[project]
dependencies = [
    "open-climate-service[server] @ git+https://github.com/dhis2/open-climate-service.git",
    "ocs-plugins @ git+https://github.com/dhis2-chap/ocs-plugins.git",
]

[tool.uv.sources]
open-climate-service = { git = "https://github.com/dhis2/open-climate-service.git", branch = "main" }
ocs-plugins = { git = "https://github.com/dhis2-chap/ocs-plugins.git", branch = "main" }
```

Keep the `override-dependencies` block from OCS's own `pyproject.toml` under `[tool.uv]` in the
instance, as with any OCS instance. Then:

```bash
uv sync
make run    # or: uv run uvicorn open_climate_service.main:app --port 9000
```

The datasets below now appear at `GET /data-sources` and on the Data sources page. Ingest one as
you would a built-in dataset:

```bash
curl -s -X POST http://localhost:9000/ingestions \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "era5land_temperature_max_daily", "start": "2025-06-01", "end": "2025-06-30"}'
```

An instance can override any template here by putting one with the same `id` in its own
`plugins_dir/rasters/`: the instance's `plugins_dir` takes precedence over installed packages,
which take precedence over the built-ins.

## Datasets

| Dataset id | Period | Source | Access |
|---|---|---|---|
| `era5land_temperature_min_daily` | daily | ERA5-Land daily statistics (Copernicus CDS) | CDS key |
| `era5land_temperature_max_daily` | daily | ERA5-Land daily statistics (Copernicus CDS) | CDS key |
| `era5land_dewpoint_daily` | daily | ERA5-Land daily statistics (Copernicus CDS) | CDS key |
| `era5land_soil_moisture_daily` | daily | ERA5-Land daily statistics, layer 1, 0–7 cm (Copernicus CDS) | CDS key |
| `era5land_wind_speed_daily` | daily | ERA5-Land hourly u/v, mean of hourly speeds (Copernicus CDS) | CDS key |
| `cams_dust_aod_daily`, `cams_pm10_daily`, `cams_pm25_daily` | daily | CAMS EAC4 reanalysis, then CAMS global forecasts (Copernicus ADS) | CDS key, ADS terms and licences accepted |
| `glofas_river_discharge_daily` | daily | GloFAS v4 historical (Copernicus EWDS) | CDS key, EWDS terms and licence accepted |
| `imerg_precipitation_daily` | daily | NASA GPM IMERG v07, Final then Late run (GES DISC) | Earthdata Login |
| `modis_flood_daily` | daily | NASA LANCE MODIS near-real-time flood (MCDWD), about the last 7 days only | Earthdata Login |
| `modis_lst_day_daily` | daily | MODIS Terra MOD11A1 v061 daytime land surface temperature, good-quality cells, from 2000 (NASA LP DAAC) | Earthdata Login |
| `modis_lst_day_monthly`, `modis_lst_night_monthly` | monthly | MODIS Terra MOD11A2 v061 day and night land surface temperature, mean of the 8-day composites weighted by their days in the month, from March 2000 (Microsoft Planetary Computer) | none |
| `modis_ndvi_monthly`, `modis_evi_monthly` | monthly | MODIS MOD13Q1/MYD13Q1 v6.1 (Microsoft Planetary Computer) | none |
| `dynamic_world_landcover_monthly` | monthly | Dynamic World 10 m land cover, most frequent class per month, from July 2015, about 110 m by default (Google Earth Engine) | Earth Engine account and project |
| `clms_ndvi_dekadal` | dekadal | Copernicus Land Monitoring Service NDVI 300 m V3, from 2014 (Copernicus Data Space Ecosystem) | CDSE S3 keys |
| `clms_gpp_dekadal`, `clms_npp_dekadal` | dekadal | Copernicus Land Monitoring Service GPP and NPP 300 m V2, from November 2013, final values only (Copernicus Data Space Ecosystem) | CDSE S3 keys |
| `esa_worldcover_2021` | yearly | ESA WorldCover 10 m land cover, 2021 (AWS Open Data) | none |
| `modis_landcover_yearly` | yearly | MODIS MCD12Q1 v061 land cover, IGBP classes, from 2001 (NASA LP DAAC) | Earthdata Login |
| `esacci_landcover_fraction_yearly`, `esacci_landcover_class_yearly` | yearly | ESA CCI / C3S land cover, 1992–2022 (Copernicus CDS) | CDS key, dataset licence accepted |
| `worldpop_agesex_global2_100m` | yearly | WorldPop Global2 population by age and sex, 100 m, 2015–2030; slow to ingest, about 40 GeoTIFFs per country-year (WorldPop) | none |
| `rwi` | yearly | Meta Relative Wealth Index, 2021, for the instance's `extent.country_code`, low- and middle-income countries only (HDX) | none |
| `copdem90_elevation_static` | static | Copernicus DEM GLO-90 (AWS Open Data); the 30 m DEM is in OCS core | none |

Each template's `description` says more about the dataset, including its known limits. Two to
note: **the MODIS flood product keeps only about the last 7 days**, so its record starts when it
is first ingested and needs a daily sync to grow; and **MODIS science collection ends in 2027**
(Terra early, Aqua late), so the MODIS datasets stop updating then.

**Relative Wealth Index values rank areas within one country only**, so they cannot be compared
between instances. HDX lists its licence as "Other", with no confirmed open terms: check them
before publishing the layer anywhere public, and cite Chi, Fang, Chatterjee & Blumenstock,
"Microestimates of wealth for all low- and middle-income countries", PNAS 2022
([doi:10.1073/pnas.2113658119](https://doi.org/10.1073/pnas.2113658119)).

## Processes

| Id | Kind | What it does |
|---|---|---|
| `aggregate_spatial_population_weighted` | openEO process | The population-weighted mean of a raster per feature and time step, `sum(value × population) / sum(population)`, with a population raster such as WorldPop as `weights` |

A population-weighted mean gives the heat or rainfall where people live rather than averaged over
empty land. It differs from OCS core's `aggregate_spatial_weighted`, which weights each cell by
how much of it the feature covers. Here a cell counts if its centre is inside the feature, so use
a population raster much finer than the smallest feature: at 1 km, a small unit can have few or
no cells. Cells with no data or no people are left out, and a feature with none left has no
value rather than 0. The population is averaged to one surface if it has several years, and
stays on its own grid: each population cell takes the climate value of the cell it lies in.

## Credentials

- **Copernicus CDS, ADS and EWDS:** one key from your [CDS profile](https://cds.climate.copernicus.eu/profile),
  in `~/.ecmwfdatastoresrc` (the file the `ecmwf-datastores` client reads, not the older
  `~/.cdsapirc`), or in the `ECMWF_DATASTORES_KEY` environment variable:

  ```
  url: https://cds.climate.copernicus.eu/api
  key: <your-key>
  ```

  The same key works on the [ADS](https://ads.atmosphere.copernicus.eu) and the
  [EWDS](https://ewds.climate.copernicus.eu), but each portal's terms of use, and the licence of
  each dataset you use, must be accepted once in the browser. Until then requests fail with 403.
- **Copernicus Data Space Ecosystem (CDSE):** free S3 keys, created at
  https://eodata-s3keysmanager.dataspace.copernicus.eu after registering at
  [dataspace.copernicus.eu](https://dataspace.copernicus.eu). Set `CDSE_S3_ACCESS_KEY` and
  `CDSE_S3_SECRET_KEY`, or add them as a `[cdse]` profile in `~/.aws/credentials`:

  ```
  [cdse]
  aws_access_key_id = <access key>
  aws_secret_access_key = <secret key>
  ```
- **NASA Earthdata:** an [Earthdata Login](https://urs.earthdata.nasa.gov) in `~/.netrc`:

  ```
  machine urs.earthdata.nasa.gov login <username> password <password>
  ```

  For IMERG, also approve the *NASA GESDISC DATA ARCHIVE* application in your Earthdata profile.
  An `EARTHDATA_TOKEN` environment variable is used instead of `~/.netrc` when set.
- **Google Earth Engine:** each instance signs up for its own Earth Engine account and registers its
  own Google Cloud project at [code.earthengine.google.com/register](https://code.earthengine.google.com/register),
  as with the CDS. Set `EARTHENGINE_PROJECT` to that project's id. The client uses the credentials from
  `earthengine authenticate`, or a service account's key file in `GOOGLE_APPLICATION_CREDENTIALS`.

  Check that the instance may use Earth Engine without charge before you register. The
  [noncommercial terms](https://earthengine.google.com/noncommercial/) exclude a government agency's
  repeated production of data products, such as a scheduled monthly sync, unless the agency is in a
  Least Developed Country (UN list) or part of an Indigenous government. Other government
  instances need a commercial Earth Engine licence. Non-profits may use it free of charge, but not
  for work a government or company pays them for. The data itself is CC-BY 4.0, so a published
  dataset can be shared with the attribution in its description.
- **Microsoft Planetary Computer** and **AWS Open Data:** no account needed.

## Develop

```bash
uv sync               # installs OCS from its main branch, with the server stack
uv run pytest         # unit tests, and a check that OCS discovers and can import every template
uv run ruff check . && uv run ruff format --check .
```

To try the plugins in a local instance, add a `climate-service.yaml` (see OCS's
`climate-service.yaml.example`) and run `uv run uvicorn open_climate_service.main:app` from this
repository: the package is installed in its own environment, so OCS discovers it.

A new plugin goes in `ocs_plugins/rasters/` as a `BaseDatasetPlugin` module with its YAML
template beside it, the template's `ingestion.plugin` naming the class by its full dotted path
(`ocs_plugins.rasters.<module>.<Class>`). Each plugin should name its source, its licence and
what access it needs. See OCS's [Adding custom datasets](https://github.com/dhis2/open-climate-service/blob/main/docs/adding_custom_datasets.md)
and [Installable plugins](https://github.com/dhis2/open-climate-service/blob/main/docs/installable_plugins.md) guides.
An openEO process goes in `ocs_plugins/processes/` as a `@process` function.

## Credits

Most of the plugins were written by Knut Rand ([@knutdrand](https://github.com/knutdrand)) for a
climate–health use case, first proposed for OCS core in dhis2/open-climate-service#459–#464.
The Copernicus Land Monitoring Service GPP and NPP plugin is based on the ones written for the
Uganda and Malawi climate service instances by Diana Nanyanzi and Yambanso. The population-weighted
aggregation follows the DHIS2 Climate Tools [population weighting guide](https://climate-tools.dhis2.org/guides/aggregation/population-weighting/).
