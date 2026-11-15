# Experiment protocol

## Prediction targets and timing

For station x, HRRR initialization t and lead h, the verification report is the
nearest station temperature within 20 minutes of t+h. Temperature is in Celsius.
The direct model predicts that temperature. The MOS model predicts
`error_c = observed_t2m_c - hrrr_t2m_c`; the final forecast is HRRR plus the
predicted residual, added exactly once.

Current observations must precede t minus a 10-minute assumed arrival margin.
History uses the latest eligible report before t−3h, t−6h and t−24h, with maximum
age 90 minutes. Satellite creation time plus a five-minute margin must precede t;
scan end must precede creation and lie within the configured lookback. The exact
selected report and image times are retained.

Forecast issue is assumed to be t+90 minutes. This is not a verified historical
publication time. Thus h is measured from initialization, not issue. IEM historical
revisions/arrival times are not fully reconstructible from this archive. These
assumptions must be disclosed in a paper; this is retrospective evaluation, not a
certified real-time simulation. Direct inputs exclude future HRRR fields.

## Inputs and imagery

Core numerical inputs: latitude, longitude, lead, cyclic initialization hour and
annual phase, current temperature and observation age. MOS also receives HRRR
2-m temperature. Optional groups are station history, HRRR U/V wind, HRRR dewpoint,
local C13, contextual C13 and C08 water vapour. HRRR optional groups are MOS-only.

Each satellite image is reprojected to a station-centered 1600 km square. Only
DQF=0 native pixels are accepted. Missing pixels remain masked. The retained grid
is 800 × 800. Each hybrid also receives a mandatory 32 × 32 C13 overview.

| Lead | Context width | Local C13 width |
|---|---:|---:|
| 6 h | 400 km | 200 km |
| 12 h | 800 km | 400 km |
| 24 h | 1200 km | 600 km |
| 48 h | 1600 km | 800 km |

Optional images are 128 × 128. Widths are experimental settings, not validated
optima. `prepare_action_crops: true` stores content-addressed, compressed crops
once during import; overlapping identical geometries share a file. This avoids
repeatedly decoding an 800 × 800 patch in every epoch. Original patches remain
available. No CNN embeddings are precomputed: image encoders are trained jointly.

## Splits and fairness

2021–2022 are training, 2023 is validation, 2024 is temporal test. IML and LBF are
held out from all fitting; only their 2024 cases form `test_spatial`. Earlier
held-out cases are retained as `heldout_unused`. Samples with verification windows
crossing training/validation boundaries are purged. No shuffled train/test split
is used.

Within the training years, the first 80% of unique initialization times train the
predictors; the last 20% train the acquisition utility models. The same boundary
purge applies. Every predictor and statistical baseline uses the same predictor
partition, including models without acquisition, to keep the amount of fitting
data comparable. Normalizers fit only that partition. Targets are unscaled;
images use the fixed transform `(kelvin − 250)/50` and explicit validity masks.

The common evaluation cohort requires a label, HRRR temperature and current
station observation. It is identical for every model, seed and budget within each
test split. Missing optional features are masked; missing targets are never filled
in. Coverage reports retain exclusions so headline metrics cannot hide missingness.

Geographic holdout tests transfer to these two stations, not independence from all
shared weather systems or nearby pixels. The architecture uses coordinates, but
arbitrary-location generalization is not established by this experiment.

## Predictors and selection

The numerical MLP uses two hidden layers of 128 and 64 SiLU units with dropout .1.
Hybrid models add small convolutional encoders (16/32/32 channels, GroupNorm,
SiLU, pooling, 16-dimensional embeddings). Numerical missingness and acquisition
indicators are explicit inputs. Separate encoders handle each image action.

Profiles:

1. `numeric_mlp`: numerical core and all permitted numerical optional groups.
2. `overview_hybrid`: the same plus the coarse mandatory overview.
3. `full_hybrid`: all numerical and image actions.
4. `masked_hybrid`: randomly selected valid acquisition sequences during training.

AdamW minimizes MSE, with gradient clipping, a maximum 60 epochs and validation
patience 10. Learning rates .001 and .0003 are tested with seeds 11, 29 and 47.
A profile/mode selects the rate with lowest mean validation MSE across all three
seeds. All three selected seeds are tested; no best-seed cherry-picking occurs.
Masked validation uses fixed reproducible masks/budgets across epochs.

Baselines are current-observation persistence, raw HRRR, training seasonal/hourly
climatology, station/lead mean HRRR bias correction, multiple linear regression,
ridge regression and a Gaussian process. Linear/ridge/GP models exist in both
direct and MOS modes and use all permitted numerical inputs with validity masks.
Ridge alpha and GP length scale are selected only on validation data. The GP is an
exact RBF+white-noise GP fitted to a fixed random subset of at most 2,000 training
cases; it is not a full-dataset GP. Climatology and bias correction fall back to
regional training estimates for unseen stations. Bias correction is static, not
a rolling operational bias estimator. External NBM comparison is not implemented.

## Budgeted acquisition

After freezing each selected masked predictor, teacher calculations on the
chronological policy-training partition measure the marginal change in squared
error from acquiring one additional group. These calculations may inspect targets
and optional features during fitting. A standardized ridge utility model per
action predicts the gain from the currently visible predictor representation and
remaining budget. The default calibration uses up to 2,000 cases and three states
per case. It is an implementable baseline policy, not an optimality guarantee.

At inference, the learned policy sees current core/overview/acquired inputs only.
It never sees labels or unacquired values/quality masks. It greedily selects the
largest predicted gain divided by cost and stops if none is positive. Static
acquisition uses mean training gain/cost. Random acquisition selects affordable
unacquired actions. All share the same frozen masked predictor and hard budget
checks. An unacquired image never executes its CNN encoder. Encoders for already
acquired images may run again during sequential decisions; policy overhead is not
included in the feature-cost proxy.

Budgets are 0, .15, 1.15, 2.15 and 3.15. Each numerical group costs .05 and each
image action costs 1. These are engineering proxies, not measured latency or FLOPs.
Mandatory numerical inputs and overview are outside the optional budget. Report
actual acquired cost, which can be below the cap. Missing attempted acquisitions
still consume their configured cost.

`source_bytes` describes hypothetical incremental distinct source files beyond
mandatory inputs for one cold-cache case. Training imports candidate features in
advance; inference does not download them again. Network cost, feature proxy cost,
CPU preprocessing and GPU latency are different quantities. Fixed-model timings
cover batched model forward only; sequential-policy timings include local IO and
policy overhead. Do not compare these timing columns as equal measurement scopes.
A hardware-calibrated cost study is additional experimental work.

## Reports and interpretation

Outputs include MAE, RMSE, mean error (prediction minus observation), MSE skill
relative to raw HRRR, optional costs, lead/station/season breakdowns, and seed
means/sample standard deviations. Every comparison uses paired sample IDs.

Overall uncertainty is a paired seven-day block bootstrap: all stations and leads
within a block are resampled together. At least eight blocks are required; otherwise
intervals are null. The interval is for raw-HRRR RMSE minus model RMSE, so positive
values favour the model. This does not adjust for multiple model comparisons.
Seed variation and sampling uncertainty are reported separately.

Keep test sets sealed until settings are selected. Inspecting test performance and
then tuning on it changes the protocol and requires a new untouched evaluation.
No numerical model-skill claims are supplied before the real archive is trained.
