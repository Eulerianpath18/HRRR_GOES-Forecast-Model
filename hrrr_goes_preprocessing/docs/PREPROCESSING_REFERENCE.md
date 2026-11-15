# Preprocessing and Dataset API reference

See ../README.md and EXPERIMENTS.md for the current complete workflow. Monthly
shards use the layout below; final tables link to their images under `shards/`.
Research imports also store small precomputed action crops under `crops/` within
each shard. These contain the same values as dynamic cropping.

## Scientific timing contract

For nominal HRRR initialization `t`, lead `h`, and station `x`, the label is a
station report near `t+h`. `error_c = observed_t2m_c - hrrr_t2m_c`, so MOS returns
`HRRR + predicted_error`. The target remains in Celsius; MOS targets are Celsius errors.

`issue_utc = t + issue_delay_minutes` (default 90 minutes). A real HRRR run cannot
be treated as available at its nominal initialization. This deadline is an explicit
assumption, **not a verified historical publication time**. All observational input
cutoffs are kept at `t`, for both direct and MOS experiments. Thus `lead_h` is measured
from initialization, and the remaining lead from the assumed issue is shorter.
The direct experiment receives station/time/location inputs and satellite data;
HRRR future predictors are restricted to MOS. Match operational comparisons to these
same availability rules, not just nominal model lead.

For current station predictors, we require report time <= `t - observation_latency_minutes`.
Each historical predictor uses the latest report no later than `t-lag`, additionally
bounded by that latency cutoff. Default maximum report age relative to its query is
90 minutes. Actual input age and historical report times are saved. IEM does not
supply immutable historical arrival/revision histories here; configurable latency is
an approximation. Archive corrections can still differ from data available in real time.

GOES must satisfy `scan_end <= creation` and `creation + margin <= t`, with scan end
inside the configured lookback. This avoids choosing the nearest future scan. Bands
are selected independently; their exact times are recorded and may differ. Use those
times when studying multiband temporal consistency. GOES-19 is used in the original August 2025 smoke config. The research
configuration uses GOES-16 for 2021–2024. Choose the proper satellite and verify its archive/coverage for other
dates; there is no silent satellite substitution. Bands 7–16 are supported in kelvin.
Visible reflectance channels require different handling and are rejected.

Only DQF=0 native pixels are accepted. Bad pixels are masked before bilinear
interpolation; missing coverage stays NaN until tensor construction, which returns a
validity mask and replaces missing normalized values by zero. The large-patch default
is 1600 km / 800 samples, roughly 2 km spacing. It cannot recover finer information
than that stored grid. The 6/12/24/48h widths of 400/800/1200/1600 km are initial
experimental settings, not established optima. Crop offsets are supported, but no
advection velocity is assumed or fitted automatically.

## Output layout and meaning

| File/directory | Contents |
|---|---|
| `samples.parquet` | Sample ID, station, init/issue/valid times, core inputs, raw HRRR temperature, labels, residual and split |
| `optional.parquet` | Sample ID and optional station-history/HRRR predictor values |
| `manifest.json` | Feature schema, allowed actions, costs, source resources, per-sample scene references and timing metadata |
| `satellite/*.npz` | Per-station/per-scene/per-band large image (`image_k`, `[1,N,N]`), footprint and coordinates |
| `satellite/*.json` | Source, timing, reprojection and valid-fraction metadata |
| `overviews/*.npz` | Small C13 context image, stored separately so reset does not read high-resolution arrays |
| `observations/*.parquet` | Cleaned station report series used for matching |
| `raw/` | Cached IEM responses, raw GOES NetCDF, HRRR GRIB subsets and extracted-point caches |
| `build_config.json` | Configuration, station metadata and fingerprint |
| `report.json`, `failures.json` | Counts, audit results and explicit acquisition failures |

Large scenes are shared across leads, and raw GOES scenes across stations. Missing
labels stay missing; the supervised Dataset excludes them. A missing optional image
retains its action slot with a false validity mask. `acquired=true, valid=false` means
an acquisition was attempted but its data are missing. With `strict: true` (default),
source failures stop the build and are recorded. With `strict: false`, missing sources
remain missing and the report exposes their coverage. Monthly reports are preliminary; the ARC workflow requires full acceptance in READY.json before training.
Do not run concurrent writers against the same output directory.

## PyTorch usage

After building a dataset with chronological splits:

```python
from torch.utils.data import DataLoader
from forecastprep.torch_data import ForecastDataset, Standardizer

root = 'data/pilot'
normalizer = Standardizer.fit(root, 'data/pilot/normalizer.json')
train = ForecastDataset(root, mode='mos', split='train', normalizer=normalizer,
                        actions=['history', 'infrared_local'], budget=1.05)
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
batch = next(iter(loader))
# batch['inputs']: core, core_mask, overview, overview_mask,
#                  optional[action_id] = value/valid/acquired, remaining_budget
# batch['target']: [batch], Celsius for direct or Celsius error for MOS
# batch['sample_id']: identifiers for joining predictions back to samples.parquet
```

Use `Standardizer.load` for validation/test; never refit on those splits. Numeric
columns use training means/stds only. Images use the fixed physical transform
`(brightness_temperature_K - 250)/50`; no held-out image statistics are fitted.
Targets are not standardized. Unrequested actions have zero tensors, false masks and
false acquisition flags. HRRR optional actions do not exist in direct-mode inputs.
Put DataLoader construction inside `if __name__ == '__main__':` on Windows when using
workers; the example scripts do this. CUDA transfer happens in your training loop.

For randomly masked predictor training, use `RandomAcquisitionDataset(root, mode,
split='train', normalizer=normalizer, budgets=(0, 1, 2, 3))`. It samples valid action
sequences within budget, using PyTorch's worker RNG. Seed PyTorch and your DataLoader
for reproducibility. Use fixed selections for deterministic validation, or evaluate
the learned policy through `AcquisitionSession`.

The smoke config has `splits: null`, so its rows are `unsplit`; use `split=None` to
inspect them. Fitting a standardizer deliberately fails if there are no training rows.

## Budgeted acquisition

```python
from forecastprep.torch_data import FeatureStore, AcquisitionSession

store = FeatureStore('data/pilot', mode='mos', normalizer=normalizer)
session = AcquisitionSession(store, store.rows.index[0], budget=1.1)
state = session.observation()           # mandatory predictors + coarse overview
choices = session.available_actions()  # IDs and configured costs, no hidden values
state = session.acquire('history')
state = session.acquire('water_vapour')
# Feed state to predictor; a policy can decide the next action from this state.
```

This is an offline research interface, not a security sandbox: training code can
access the underlying store for oracle targets. The *policy observation* contains
neither labels nor unacquired values. A value network must receive only this observation
and action metadata, never `store.optional` or supervised targets at inference.
The current experiment suite trains the masked predictor first, then estimates
value-of-acquisition targets on a separate chronological training holdout.

`cost_units` are explicit positive engineering proxies. Budget covers **optional
compute**, with mandatory inputs treated as a common fixed baseline across policies.
A 128x128 single-band patch defaults to 1 unit and a small tabular group to 0.05;
these are not measured latency or FLOPs. Calibrate them on the actual encoders/device
before making computational-efficiency claims. Different patch sizes require adjusted
costs. Acquiring an action twice or exceeding the budget raises an error.

`source_bytes` separately tracks incremental unique GOES/GRIB resources needed after
mandatory sources; the coarse overview already pays for its full C13 file. Therefore
another C13 crop may cost zero network bytes while still costing computation. IEM
history reuses the mandatory observation table. Archive ingestion downloads candidate
features upfront for policy training; replayed source bytes are a hypothetical cold-cache
retrieval cost, **not new bytes downloaded during replay**. Share `paid_resources=set()`
across sequential sessions to model cross-station/cross-lead source reuse. Unknown byte
sizes are exposed in `unknown_size_resources`, never guessed. `load_seconds` records actual local load/crop/tensor time for each acquired action;
it excludes CNN execution. This does not measure
GPU time, disk I/O, RAM, model training cost, or wall-clock network latency.

## Splits and comparison design

Samples whose label window crosses a training/validation boundary are marked `purged`.
Optional embargo widens the separation. Station IDs in `heldout_stations` are excluded
from train/validation; their post-test-boundary rows become `test_spatial`, earlier rows
`heldout_unused`. Apply the same held-out targets to direct and MOS comparisons.
`test` uses stations seen in training; `test_spatial` tests temporal and spatial transfer.
Training-only normalization cannot see heldout-station values.

Nearby sites can share pixels and weather systems, so station holdouts alone do not
ensure spatial independence; choose geographically meaningful holdout blocks for the
paper. Station exposure/elevation versus nearest HRRR cell also creates representativeness
error. Grid coordinates/distance are retained; elevation adjustment is not implemented.

The current experiment suite implements persistence, climatology, subset GP,
linear/ridge models, static bias MOS and numerical/satellite neural models.
External operational NBM ingestion remains outside this implementation.

## References used for the implementation

- Herbie API: https://herbie.readthedocs.io/en/stable/api_reference/_autosummary/herbie.core.Herbie.html
- IEM API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
- NOAA GOES projection: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/satellite_data_goes_imager_projection.php
- NOAA product descriptions: https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm
- PyTorch datasets/loaders: https://docs.pytorch.org/docs/stable/data.html

The package includes `example_data/denver_smoke`, the processed real smoke run
(two rows and the satellite arrays, with source metadata). You can inspect it without
network access:

```powershell
forecastprep audit example_data/denver_smoke
python examples/acquire.py example_data/denver_smoke
```

This is not enough data to train a useful model. Raw GRIB/NetCDF caches are omitted
from the bundled example to keep the download small; `build` downloads them normally.
The recorded configuration paths are provenance from the build machine, not paths
needed to read the bundled dataset.

See `VALIDATION.md` for the actual checks performed and their limits.
